"""Generate the nine figures specified in the manuscript figure plan.

Run from the repository root::

    python -m src.figures \
        --figures 2-10 \
        --output-dir output/figures \
        --formats pdf,png \
        --K_Q 66.0 \
        --k_b 0.2 \
        --Q_max 1.0 \
        --n_b 1.0 \
        --grid-size 40 \
        --sobol-base 128 \
        --jobs 6

For the manuscript-resolution analyses, use ``--grid-size 80`` and
``--sobol-base 1024``.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/acptohapmodel-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use(os.environ["MPLBACKEND"], force=True)
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np

from src.analysis import evaluate_parameter_vector, inventory_error, ordered_map, output_metrics, parameter_dict, save_kq_ka_ratio_sweep
from src.model import (
    DEFAULT_OUTPUT_DIR,
    EPS,
    MineralizationModel,
    ModelParameters,
    SimulationResult,
    SimulationSettings,
    add_model_arguments,
    add_stochastic_arguments,
    parameters_from_args,
    resolve_output_path,
    settings_from_args,
    simulate_ensemble,
    stochastic_from_args,
)


FIGURE_STEMS = {
    2: "fig02_baseline_dynamics",
    3: "fig03_pathway_regime_map",
    4: "fig04_pathway_competition",
    5: "fig05_protocol_pathway_composition",
    6: "fig06_dynamic_parameter_sweeps",
    7: "fig07_sobol_matrix_and_bulk",
    8: "fig08_identifiability",
    9: "fig09_literature_comparison",
    10: "fig10_numerical_verification",
    11: "fig11_KQ_KA_ratio_sensitivity",
}

COLORS = {
    "B": "#7A5195", "C": "#0072B2", "P": "#E69F00", "A": "#D55E00",
    "H": "#009E73", "Q": "#CC79A7", "total": "#222222", "matrix": "#56B4E9",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, number: int, formats: tuple[str, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(output_dir / f"{FIGURE_STEMS[number]}.{extension}", bbox_inches="tight")
    plt.close(fig)


def save_named_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: tuple[str, ...]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold")


def add_renewals(ax: plt.Axes, settings: SimulationSettings) -> None:
    for day in np.arange(settings.media_interval, settings.t_end, settings.media_interval):
        ax.axvline(day, color="0.85", linewidth=0.55, zorder=0)


def figure_02(output_dir: Path, formats: tuple[str, ...], parameters: ModelParameters, settings: SimulationSettings, ensemble_runs: list[SimulationResult] | None = None, **_: object) -> None:
    result = MineralizationModel(parameters).simulate(settings)
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 9.0), sharex=True)
    axes[0].plot(result.t, result.C, color=COLORS["C"], label="Calcium C")
    axes[0].plot(result.t, result.P, color=COLORS["P"], label="Phosphate P")
    if ensemble_runs:
        for field, color in (("C", COLORS["C"]), ("P", COLORS["P"])):
            values = np.asarray([getattr(run, field) for run in ensemble_runs])
            axes[0].fill_between(result.t, *np.quantile(values, [0.025, 0.975], axis=0), color=color, alpha=0.13)
    axes[0].set_ylabel("Soluble ion (mM)")
    axes[0].set_title("Soluble calcium and phosphate")
    axes[0].legend(ncol=2)
    axes[1].plot(result.t, result.S_A, color="#3B528B", label="$S_A$")
    axes[1].plot(result.t, result.S_Q, color=COLORS["Q"], linestyle="-", alpha=0.75, label="$S_Q$")
    axes[1].axhline(1.0, color=COLORS["A"], linestyle="--", label="$S_A=1$")
    axes[1].axhline(1.0, color=COLORS["Q"], linestyle=":", label="$S_Q=1$")
    axes[1].set_ylabel("Formation index")
    axes[1].set_title("Pathway activation indices")
    axes[1].legend(ncol=1)
    for name in ("A", "H", "Q"):
        axes[2].plot(result.t, getattr(result, name), color=COLORS[name], label=name)
        if ensemble_runs:
            values = np.asarray([getattr(run, name) for run in ensemble_runs])
            axes[2].fill_between(result.t, *np.quantile(values, [0.025, 0.975], axis=0), color=COLORS[name], alpha=0.13)
    axes[2].set_ylabel("Retained pool (mM P-eq.)")
    axes[2].set_title("Retained mineral pools")
    axes[2].legend(ncol=3)
    axes[3].plot(result.t, result.M_tot, color=COLORS["total"], label="$M_{tot}=A+H+Q$")
    axes[3].plot(result.t, result.M_mat, color=COLORS["matrix"], label="$M_{mat}=A+H$")
    if ensemble_runs:
        for field, color in (("M_tot", COLORS["total"]), ("M_mat", COLORS["matrix"])):
            values = np.asarray([getattr(run, field) for run in ensemble_runs])
            axes[3].fill_between(result.t, *np.quantile(values, [0.025, 0.975], axis=0), color=color, alpha=0.13)
    axes[3].set_ylabel("Retained mineral\n(mM P-eq.)")
    axes[3].set_xlabel("Time (day)")
    axes[3].set_title("Matrix-associated and total mineral")
    axes[3].legend(ncol=2)
    for label, ax in zip("ABCD", axes):
        panel_label(ax, label)
        add_renewals(ax, settings)
    fig.suptitle("Baseline soluble-ion and retained-mineral dynamics")
    fig.tight_layout()
    save_figure(fig, output_dir, 2, formats)
    result.to_csv(output_dir / "fig02_baseline_trajectory.csv")


def analytical_crossing(parameters: ModelParameters, beta_gp: float, pathway: str) -> float:
    model = MineralizationModel(parameters.changed(B_ext=beta_gp))
    return model.threshold_times()[0 if pathway == "A" else 1]


def figure_03(output_dir: Path, formats: tuple[str, ...], parameters: ModelParameters, settings: SimulationSettings, grid_size: int = 50, **_: object) -> None:
    beta_values = np.linspace(0.0, 15.0, 151)
    intervals = np.linspace(0.25, 6.0, 140)
    t_a = np.array([analytical_crossing(parameters, beta, "A") for beta in beta_values])
    t_q = np.array([analytical_crossing(parameters, beta, "Q") for beta in beta_values])
    regime = np.zeros((intervals.size, beta_values.size))
    regime[intervals[:, None] > t_a[None, :]] = 1
    regime[intervals[:, None] > t_q[None, :]] = 2

    endpoint_beta = np.linspace(beta_values[0], beta_values[-1], max(25, min(grid_size + 1, 41)))
    endpoint_intervals = np.linspace(intervals[0], intervals[-1], max(25, min(grid_size, 40)))
    analytical_regime = np.zeros((endpoint_intervals.size, endpoint_beta.size))
    endpoint_t_a = np.array([analytical_crossing(parameters, beta, "A") for beta in endpoint_beta])
    endpoint_t_q = np.array([analytical_crossing(parameters, beta, "Q") for beta in endpoint_beta])
    analytical_regime[endpoint_intervals[:, None] > endpoint_t_a[None, :]] = 1
    analytical_regime[endpoint_intervals[:, None] > endpoint_t_q[None, :]] = 2
    mineral_floor = 1.0e-3
    matrix_fraction = np.full((endpoint_intervals.size, endpoint_beta.size), np.nan)
    matrix_endpoint = np.zeros_like(matrix_fraction)
    solution_endpoint = np.zeros_like(matrix_fraction)
    total_endpoint = np.zeros_like(matrix_fraction)
    for i, interval in enumerate(endpoint_intervals):
        run_settings = SimulationSettings(settings.t_end, settings.dt, float(interval), settings.method, settings.rtol, settings.atol)
        for j, beta_gp in enumerate(endpoint_beta):
            result = MineralizationModel(parameters.changed(B_ext=float(beta_gp))).simulate(run_settings)
            matrix_endpoint[i, j] = result.M_mat[-1]
            solution_endpoint[i, j] = result.Q[-1]
            total_endpoint[i, j] = result.M_tot[-1]
            if total_endpoint[i, j] > mineral_floor:
                matrix_fraction[i, j] = matrix_endpoint[i, j] / total_endpoint[i, j]

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), sharex=True, sharey=True)
    regime_cmap = ListedColormap(["#242424", "#4C78A8", "#8E5EA2"])
    axes[0].pcolormesh(beta_values, intervals, regime, cmap=regime_cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), shading="auto")
    handles = [
        plt.Line2D([], [], marker="s", linestyle="", color=regime_cmap(i), markersize=9, label=label)
        for i, label in enumerate(("No mineral", "Matrix-selective", "Both pathways"))
    ]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.99), frameon=True, framealpha=0.96, facecolor="white", edgecolor="0.75")

    fraction_cmap = plt.get_cmap("magma").copy()
    fraction_cmap.set_bad("#242424")
    im1 = axes[1].pcolormesh(endpoint_beta, endpoint_intervals, np.ma.masked_invalid(matrix_fraction), cmap=fraction_cmap, vmin=0.0, vmax=1.0, shading="auto")
    finite_fraction = np.isfinite(matrix_fraction)
    if np.any(finite_fraction) and np.nanmin(matrix_fraction) <= 0.5 <= np.nanmax(matrix_fraction):
        axes[1].contour(endpoint_beta, endpoint_intervals, matrix_fraction, levels=[0.5], colors="#303030", linewidths=1.0)
    fig.colorbar(im1, ax=axes[1], label="Matrix-associated fraction, $(A+H)/(A+H+Q)$")
    for ax in axes:
        ax.scatter(parameters.B_ext, settings.media_interval, marker="*", s=130, c="white", edgecolors="black", zorder=5)
        ax.set_xlabel("β-GP (mM)")
    axes[0].set_ylabel("Medium-renewal interval (day)")
    axes[0].set_title("Pathway regime")
    axes[1].set_title("Final matrix-associated mineral fraction")
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    fig.tight_layout()
    save_figure(fig, output_dir, 3, formats)
    with (output_dir / "fig03_final_matrix_originated_fraction.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["beta_GP", "media_interval", "matrix_originated_fraction", "M_mat_final", "Q_final", "M_tot_final"])
        for i, interval in enumerate(endpoint_intervals):
            for j, beta_gp in enumerate(endpoint_beta):
                writer.writerow([beta_gp, interval, matrix_fraction[i, j], matrix_endpoint[i, j], solution_endpoint[i, j], total_endpoint[i, j]])


def figure_04(output_dir: Path, formats: tuple[str, ...], parameters: ModelParameters, settings: SimulationSettings, **_: object) -> None:
    bulk_threshold_on_s_a = parameters.K_Q / parameters.K_A
    formation_index = np.linspace(0.05, max(6.0, bulk_threshold_on_s_a + 4.0), 240)
    occupancy = np.linspace(0.0, 0.99, 200)
    S_grid, F = np.meshgrid(formation_index, occupancy)
    capacity = 1.0 - F
    rf = parameters.k_f * np.maximum(S_grid - 1.0, 0.0) * parameters.M_max * capacity
    S_Q_grid = S_grid * parameters.K_A / parameters.K_Q
    bulk_capacity = max(parameters.Q_max, EPS)
    rb = parameters.k_b * np.maximum(S_Q_grid - 1.0, 0.0) ** parameters.n_b * bulk_capacity
    competition = np.divide(rb, rf, out=np.zeros_like(rb), where=rf > 0.0)
    phi = np.divide(rf, rf + rb, out=np.ones_like(rf), where=(rf + rb) > 0.0)

    representative_settings = SimulationSettings(
        settings.t_end, settings.dt, max(4.0, settings.media_interval), settings.method, settings.rtol, settings.atol
    )
    representative = MineralizationModel(parameters.changed(B_ext=max(20.0, parameters.B_ext))).simulate(representative_settings)
    trajectory_occupancy = representative.M_mat / representative.parameters.M_max
    keep = representative.reset_status != "pre_reset"
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharex=True, sharey=True)
    log_comp = np.log10(np.maximum(competition, 1.0e-4))
    im0 = axes[0].pcolormesh(S_grid, F, log_comp, cmap="coolwarm", vmin=-2, vmax=2, shading="auto")
    axes[0].contour(S_grid, F, competition, levels=[1.0], colors="black", linewidths=1.4)
    fig.colorbar(im0, ax=axes[0], label=r"$\log_{10}(R_b/R_f)$, solution-driven vs matrix-associated")
    im1 = axes[1].pcolormesh(S_grid, F, phi, cmap="viridis", vmin=0, vmax=1, shading="auto")
    axes[1].contour(S_grid, F, phi, levels=[0.1, 0.25, 0.5, 0.75, 0.9], colors="white", linewidths=0.7)
    fig.colorbar(im1, ax=axes[1], label="$Φ_{mat}$, matrix-associated fraction")
    color_values = representative.t[keep]
    path = axes[2].scatter(representative.S_A[keep], trajectory_occupancy[keep], c=color_values, cmap="plasma", s=5)
    fig.colorbar(path, ax=axes[2], label="Time (day)")
    axes[2].axvline(bulk_threshold_on_s_a, color="0.3", linestyle=":")
    titles = ("Pathway competition", "Instantaneous matrix-associated fraction", "Baseline trajectory through pathway space")
    for label, title, ax in zip("ABC", titles, axes):
        panel_label(ax, label)
        ax.set_title(title)
        ax.set_xlabel("Formation index $S_A$")
        ax.set_xlim(formation_index[0], formation_index[-1])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Occupied matrix capacity $M_{mat}/M_{mat,max}$")
    fig.suptitle("Pathway competition and matrix contribution")
    fig.tight_layout()
    save_figure(fig, output_dir, 4, formats)


def protocol_results(parameters: ModelParameters, settings: SimulationSettings) -> tuple[list[str], list[SimulationResult]]:
    specifications = [
        ("Low β-GP", 5.0, settings.media_interval),
        ("Baseline", parameters.B_ext, settings.media_interval),
        ("High β-GP", 20.0, settings.media_interval),
        ("Short renewal", parameters.B_ext, 1.0),
        ("Long renewal", parameters.B_ext, 4.0),
    ]
    labels, results = [], []
    for label, beta_gp, interval in specifications:
        labels.append(label)
        results.append(MineralizationModel(parameters.changed(B_ext=beta_gp)).simulate(SimulationSettings(settings.t_end, settings.dt, interval, settings.method, settings.rtol, settings.atol)))
    return labels, results


def figure_05(output_dir: Path, formats: tuple[str, ...], parameters: ModelParameters, settings: SimulationSettings, **_: object) -> None:
    labels, results = protocol_results(parameters, settings)
    values = [
        [result.M_tot[-1] for result in results],
        [result.M_mat[-1] for result in results],
        [result.Q[-1] for result in results],
        [result.final_matrix_fraction for result in results],
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    titles = ("Final total retained mineral", "Final matrix-associated mineral", "Final solution-driven mineral", "Matrix-associated fraction of retained mineral")
    colors = (COLORS["total"], COLORS["matrix"], COLORS["Q"], COLORS["H"])
    for label, ax, data, title, color in zip("ABCD", axes.flat, values, titles, colors):
        ax.bar(np.arange(len(labels)), data, color=color, alpha=0.85)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
        ax.set_title(title)
        panel_label(ax, label)
    for ax in axes.flat[:3]:
        ax.set_ylabel("mM phosphate equivalents")
    axes.flat[3].set_ylabel("$(A+H)/(A+H+Q)$")
    axes.flat[3].set_ylim(0, 1.05)
    fig.tight_layout()
    save_figure(fig, output_dir, 5, formats)


def valid_parameter_sweep_levels(parameters: ModelParameters, name: str, levels: list[float]) -> list[float]:
    valid_levels: list[float] = []
    for value in levels:
        try:
            parameters.changed(**{name: value}).validate()
        except ValueError:
            continue
        valid_levels.append(value)
    return valid_levels


def figure_06(output_dir: Path, formats: tuple[str, ...], parameters: ModelParameters, settings: SimulationSettings, **_: object) -> None:
    sweeps = [
        ("B_ext", [0, 2.5, 5, 7.5, 10, 15, 20], "β-GP (mM)"),
        ("eta_48h", [0.05, 0.10, 0.18, 0.22, 0.33, 0.50, 0.65], "$η_{48h}$"),
        ("K_A", [24, 28, 33, 38, 44, 48], "$K_A$"),
        ("k_f", [0.05, 0.10, 0.20, 0.30, 0.50, 0.75, 1.00], "$k_f$"),
        ("M_max", [0.25, 0.5, 1, 2, 4, 8], "$M_{mat,max}$"),
        ("k_m", [0.03, 0.10, 0.20, 0.30, 0.75, 1.50, 3.00], "$k_m$"),
        ("K_Q", [50, 58, 66, 78, 90, 110, 132], "$K_Q$"),
        ("Q_max", [0.01, 0.03, 0.10, 0.30, 1.00, 2.00, 4.00], "$Q_{max}$"),
        ("k_b", [0.03, 0.10, 0.20, 0.30, 0.75, 1.50, 3.00], "$k_b$"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12.0, 10.2), sharex=True)
    sweep_rows: list[list[float | str]] = []
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for label, ax, (name, levels, title) in zip(labels, axes.flat, sweeps):
        valid_levels = valid_parameter_sweep_levels(parameters, name, levels)
        if not valid_levels:
            raise ValueError(f"No valid figure 6 sweep levels for {name} with the current baseline parameters")
        colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(valid_levels)))
        for value, color in zip(valid_levels, colors):
            result = MineralizationModel(parameters.changed(**{name: value})).simulate(settings)
            ax.plot(result.t, result.M_mat, color=color, label=f"{value:g}")
            sweep_rows.append([name, value, result.A[-1], result.H[-1], result.Q[-1], result.M_mat[-1], result.M_tot[-1], result.final_matrix_fraction, np.max(result.S_A)])
        ax.set_title(title)
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("$M_{mat}=A+H$ (mM P-eq.)")
        ax.legend(title=title)
        panel_label(ax, label)
    fig.suptitle("Parameter effects on matrix-associated mineral accumulation")
    fig.tight_layout()
    save_figure(fig, output_dir, 6, formats)
    with (output_dir / "fig06_parameter_sweep_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", "A_final", "H_final", "Q_final", "M_mat_final", "M_tot_final", "retained_matrix_fraction", "maximum_S_A"])
        writer.writerows(sweep_rows)


def figure_07(output_dir: Path, formats: tuple[str, ...], parameters: ModelParameters, settings: SimulationSettings, sobol_base: int, jobs: int, seed: int, **_: object) -> None:
    from SALib.analyze import sobol as sobol_analyze
    from SALib.sample import sobol as sobol_sample

    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_stem in ("fig07_sobol_top_terminal_H", "fig07_sobol_matrix_and_bulk"):
        for stale_file in output_dir.glob(f"{stale_stem}.*"):
            stale_file.unlink()

    names = ["beta_GP", "eta_48h", "K_A", "K_Q", "k_f", "M_max", "k_m", "k_b", "Q_max"]
    display_names = {
        "beta_GP": "β-GP",
        "eta_48h": "$η_{48h}$",
        "K_A": "$K_A$",
        "K_Q": "$K_Q$",
        "k_f": "$k_f$",
        "M_max": "$M_{mat,max}$",
        "k_m": "$k_m$",
        "k_b": "$k_b$",
        "Q_max": "$Q_{max}$",
    }
    problem = {
        "num_vars": len(names),
        "names": names,
        "bounds": [[5.0, 20.0], [0.10, 0.50], [24.0, 48.0], [50.0, 132.0], [0.05, 1.00], [0.25, 4.0], [0.03, 3.0], [0.03, 3.0], [0.01, 4.0]],
    }
    samples = sobol_sample.sample(problem, sobol_base, calc_second_order=False, scramble=True, seed=seed)
    # SALib 1.5.2 passes names directly to ``pandas.unique`` during analysis;
    # pandas 3.x requires an array-like object rather than a plain list.
    analysis_problem = dict(problem, names=np.asarray(names, dtype=object))
    base = parameter_dict(parameters)
    tasks = []
    for row in samples:
        values = dict(zip(names, row))
        changed = dict(base)
        changed.update(
            B_ext=values["beta_GP"], eta_48h=values["eta_48h"], K_A=values["K_A"],
            K_Q=values["K_Q"], k_f=values["k_f"], M_max=values["M_max"], k_m=values["k_m"],
            k_b=values["k_b"], Q_max=values["Q_max"],
        )
        run_settings = SimulationSettings(settings.t_end, settings.dt, settings.media_interval, settings.method, settings.rtol, settings.atol)
        tasks.append((changed, run_settings))
    metrics = ordered_map(evaluate_parameter_vector, tasks, jobs)
    output_names = ["terminal M_mat", "terminal Q"]
    sobol_rows: list[dict[str, float | str]] = []
    for row_index, output_name in enumerate(output_names):
        values = np.array([metric[output_name] for metric in metrics])
        if not np.all(np.isfinite(values)):
            finite_max = np.nanmax(values[np.isfinite(values)])
            values = np.nan_to_num(values, nan=finite_max, posinf=finite_max, neginf=0.0)
        if float(np.nanstd(values)) <= EPS:
            indices = {
                "S1": np.zeros(len(names)),
                "S1_conf": np.zeros(len(names)),
                "ST": np.zeros(len(names)),
                "ST_conf": np.zeros(len(names)),
            }
        else:
            try:
                indices = sobol_analyze.analyze(analysis_problem, values, calc_second_order=False, print_to_console=False)
            except (TypeError, ValueError):
                indices = {
                    "S1": np.zeros(len(names)),
                    "S1_conf": np.zeros(len(names)),
                    "ST": np.zeros(len(names)),
                    "ST_conf": np.zeros(len(names)),
                }
        indices = {name: np.nan_to_num(np.asarray(value, dtype=float), nan=0.0, posinf=0.0, neginf=0.0) for name, value in indices.items() if name in {"S1", "S1_conf", "ST", "ST_conf"}}
        for parameter, s1, s1_conf, st, st_conf in zip(names, indices["S1"], indices["S1_conf"], indices["ST"], indices["ST_conf"]):
            sobol_rows.append(
                {
                    "output": output_name,
                    "parameter": parameter,
                    "S1": float(s1),
                    "S1_conf": float(s1_conf),
                    "ST": float(st),
                    "ST_conf": float(st_conf),
                    "ST_minus_S1": float(st - s1),
                }
            )

    with (output_dir / "fig07_sobol_total_indices.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["output", "parameter", "S1", "S1_conf", "ST", "ST_conf", "ST_minus_S1"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sobol_rows)

    targets = [
        ("terminal M_mat", "fig07a_sobol_matrix_associated", "Global sensitivity of final matrix-associated mineral"),
        ("terminal Q", "fig07b_sobol_bulk_retained", "Global sensitivity of final solution-driven mineral"),
    ]
    for target_name, stem, title in targets:
        target_rows = [row for row in sobol_rows if row["output"] == target_name]
        target_rows.sort(key=lambda row: row["ST"], reverse=True)
        plot_rows = target_rows[:5]
        x = np.arange(len(plot_rows))
        s1 = np.asarray([max(float(row["S1"]), 0.0) for row in plot_rows])
        st = np.asarray([max(float(row["ST"]), 0.0) for row in plot_rows])
        interaction = np.maximum(st - s1, 0.0)
        s1_conf = np.asarray([float(row["S1_conf"]) for row in plot_rows])
        st_conf = np.asarray([float(row["ST_conf"]) for row in plot_rows])

        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        ax.bar(x, s1, color="#4C78A8", label="First-order $S_1$")
        ax.bar(x, interaction, bottom=s1, color="#ECA400", label="Interaction contribution, $S_T-S_1$")
        ax.errorbar(x, s1, yerr=s1_conf, fmt="none", ecolor="#00A878", elinewidth=1.2, capsize=3, label="$S_1$ 95% CI")
        ax.errorbar(x, st, yerr=st_conf, fmt="none", ecolor="#B07CC6", elinewidth=1.2, capsize=3, linestyle="--", label="$S_T$ 95% CI")
        ax.set_xticks(x, [display_names[str(row["parameter"])] for row in plot_rows], rotation=25, ha="right")
        ax.set_ylabel("Sobol sensitivity index")
        ax.set_ylim(0, max(1.0, float(np.nanmax(st + st_conf)) * 1.12))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right")
        fig.tight_layout()
        save_named_figure(fig, output_dir, stem, formats)

        with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
            fieldnames = ["rank", "parameter", "display_name", "S1", "S1_conf", "ST", "ST_conf", "ST_minus_S1"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for rank, row in enumerate(plot_rows, start=1):
                writer.writerow({"rank": rank, "display_name": display_names[str(row["parameter"])], **{name: row[name] for name in fieldnames if name in row}})


def identifiability_task(task: tuple[dict[str, float], SimulationSettings, np.ndarray]) -> np.ndarray:
    values, settings, sample_days = task
    result = MineralizationModel(ModelParameters(**values)).simulate(settings)
    return result.values_at(sample_days, "M_tot")


def figure_08(output_dir: Path, formats: tuple[str, ...], parameters: ModelParameters, settings: SimulationSettings, grid_size: int, jobs: int, **_: object) -> None:
    sample_days = np.array([4, 7, 10, 14, 21, 28], dtype=float)
    baseline = MineralizationModel(parameters).simulate(settings)
    target = baseline.values_at(sample_days, "M_tot")
    profile_specs = [
        ("k_f", "$k_f$", np.logspace(-2, 1, grid_size)),
        ("k_m", "$k_m$", np.logspace(-2, 1, grid_size)),
        ("k_b", "$k_b$", np.logspace(-2, 1, grid_size)),
        ("Q_max", "$Q_{max}$", np.logspace(-2, 1, grid_size)),
        ("M_max", "$M_{mat,max}$", np.logspace(-1, 1, grid_size)),
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, len(profile_specs)))
    rows: list[list[float | str]] = []
    profiles: list[dict[str, object]] = []
    for (name, display, values), color in zip(profile_specs, colors):
        misfit, final_matrix, final_solution, multipliers = [], [], [], []
        baseline_value = float(getattr(parameters, name))
        for value in values:
            candidate = MineralizationModel(parameters.changed(**{name: float(value)})).simulate(settings)
            prediction = candidate.values_at(sample_days, "M_tot")
            error = np.mean(((prediction - target) / np.maximum(target, 0.05 * target[-1])) ** 2)
            misfit.append(error)
            final_matrix.append(candidate.M_mat[-1])
            final_solution.append(candidate.Q[-1])
            multipliers.append(float(value) / baseline_value)
            rows.append([name, value, float(value) / baseline_value, error, candidate.M_mat[-1], candidate.Q[-1], candidate.M_tot[-1]])
        profiles.append({
            "display": display,
            "color": color,
            "multiplier": np.asarray(multipliers, dtype=float),
            "misfit": np.asarray(misfit, dtype=float),
            "M_mat": np.asarray(final_matrix, dtype=float),
            "Q": np.asarray(final_solution, dtype=float),
        })

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
    for profile in profiles:
        color = profile["color"]
        label = str(profile["display"])
        axes[0].plot(profile["multiplier"], np.log10(np.maximum(profile["misfit"], 1.0e-10)), color=color, label=label)
        axes[1].plot(profile["multiplier"], profile["M_mat"], color=color, label=label)
        axes[2].plot(profile["multiplier"], profile["Q"], color=color, label=label)
    for ax in axes:
        ax.axvline(1.0, color="0.25", linestyle="--", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("Parameter / baseline")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(r"$\log_{10}$ trajectory error")
    axes[1].set_ylabel("$A+H$ (mM P-eq.)")
    axes[2].set_ylabel("$Q$ (mM P-eq.)")
    axes[0].set_title("Trajectory-profile error")
    axes[1].set_title("Final matrix-associated mineral")
    axes[2].set_title("Final solution-driven mineral")
    axes[0].legend(title="Parameter", frameon=True)
    for label, ax in zip("ABC", axes):
        panel_label(ax, label)
    fig.suptitle("Practical identifiability of pathway parameters")
    fig.tight_layout()
    save_figure(fig, output_dir, 8, formats)
    with (output_dir / "fig08_identifiability_profiles.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["parameter", "value", "relative_to_baseline", "trajectory_error", "M_mat_final", "Q_final", "M_tot_final"])
        writer.writerows(rows)


def figure_09(output_dir: Path, formats: tuple[str, ...], parameters: ModelParameters, settings: SimulationSettings, **_: object) -> None:
    ars_days = np.array([4, 7, 10, 14, 21, 28], dtype=float)
    ars_mean = np.array([2, 4, 5, 12, 15, 30], dtype=float)
    ars_se = np.array([2, 3, 3, 9, 8, 6], dtype=float)
    baseline = MineralizationModel(parameters).simulate(settings)
    keep = baseline.reset_status != "pre_reset"
    model_ars = np.interp(ars_days, baseline.t[keep], baseline.M_tot[keep])
    ars_scale = ars_mean[-1] / model_ars[-1]
    scaled_ars = ars_scale * model_ars

    raman_days = np.array([0, 3, 9, 15, 24], dtype=float)
    raman_mean = np.array([0.10, 0.08, 0.40, 1.50, 4.00], dtype=float)
    raman_se = np.array([0.01, 0.01, 0.04, 0.15, 0.70], dtype=float)
    raman_result = MineralizationModel(parameters).simulate(SimulationSettings(max(settings.t_end, 24), settings.dt, 3.0, settings.method, settings.rtol, settings.atol))
    model_h = raman_result.values_at(raman_days, "H")
    model_h_norm = model_h / model_h[-1]
    raman_norm = raman_mean / raman_mean[-1]
    raman_se_norm = raman_se / raman_mean[-1]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].errorbar(ars_days, ars_mean, yerr=ars_se, fmt="o", color=COLORS["P"], label="Published ARS data")
    axes[0].plot(ars_days, scaled_ars, "-s", color=COLORS["C"], label="Model total retained mineral")
    axes[0].set_ylabel("ARS signal (a.u.)")
    axes[0].set_xlabel("Time (day)")
    axes[0].legend()
    axes[1].errorbar(raman_days, raman_norm, yerr=raman_se_norm, fmt="o", color=COLORS["Q"], label="Published Raman data")
    axes[1].plot(raman_days, model_h_norm, "-s", color=COLORS["H"], label="Model matrix-associated apatite")
    axes[1].set_ylabel("Normalized signal")
    axes[1].set_xlabel("Time (day)")
    axes[1].legend()
    for label, ax in zip("AB", axes):
        panel_label(ax, label)
    axes[0].set_title("Alizarin Red mineralization")
    axes[1].set_title("Raman apatite signal")
    fig.suptitle("Comparison with published mineralization time courses")
    fig.tight_layout()
    save_figure(fig, output_dir, 9, formats)


def figure_10(output_dir: Path, formats: tuple[str, ...], parameters: ModelParameters, settings: SimulationSettings, **_: object) -> None:
    baseline = MineralizationModel(parameters).simulate(settings)
    tight_settings = SimulationSettings(settings.t_end, settings.dt / 2.0, settings.media_interval, settings.method, settings.rtol / 100.0, settings.atol / 100.0)
    tight = MineralizationModel(parameters).simulate(tight_settings)
    p_error = inventory_error(baseline, "phosphate_inventory")
    c_error = inventory_error(baseline, "calcium_inventory")
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    axes[0, 0].plot(baseline.t, baseline.phosphate_inventory, color=COLORS["P"])
    axes[0, 0].set_ylabel("$P_{tot}$ (mM P-eq.)")
    axes[0, 1].plot(baseline.t, baseline.calcium_inventory, color=COLORS["C"])
    axes[0, 1].set_ylabel("$C_{tot}$ (mM Ca-eq.)")
    axes[1, 0].semilogy(baseline.t, np.maximum(p_error, 1e-16), color=COLORS["P"], label="Phosphate")
    axes[1, 0].semilogy(baseline.t, np.maximum(c_error, 1e-16), color=COLORS["C"], label="Calcium")
    axes[1, 0].set_ylabel("Relative within-cycle error")
    axes[1, 0].legend()
    axes[1, 1].plot(baseline.t, baseline.M_tot, color="black", label="Baseline tolerances")
    axes[1, 1].plot(tight.t, tight.M_tot, color=COLORS["Q"], linestyle="--", label="Tighter tolerances")
    axes[1, 1].set_ylabel("$M_{tot}$ (mM P-eq.)")
    axes[1, 1].legend()
    for label, ax in zip("ABCD", axes.flat):
        panel_label(ax, label)
        ax.set_xlabel("Time (day)")
        add_renewals(ax, settings)
    axes[0, 0].set_title("Phosphate-equivalent inventory")
    axes[0, 1].set_title("Calcium-equivalent inventory")
    axes[1, 0].set_title("Conservation between renewals")
    axes[1, 1].set_title("Solver-tolerance convergence")
    fig.tight_layout()
    save_figure(fig, output_dir, 10, formats)


def figure_11(output_dir: Path, formats: tuple[str, ...], parameters: ModelParameters, settings: SimulationSettings, **_: object) -> None:
    analysis_dir = output_dir.parent / "analysis"
    rows = save_kq_ka_ratio_sweep(parameters, settings, analysis_dir)
    simulated = [row for row in rows if row["status"] == "simulated"]
    ratios = np.asarray([row["KQ_KA_ratio"] for row in simulated], dtype=float)
    M_mat = np.asarray([row["final_M_mat"] for row in simulated], dtype=float)
    Q = np.asarray([row["final_Q"] for row in simulated], dtype=float)
    matrix_fraction = np.asarray([row["final_matrix_fraction"] for row in simulated], dtype=float)
    t_A = np.asarray([row["t_A_day"] for row in simulated], dtype=float)
    t_Q = np.asarray([row["t_Q_day"] for row in simulated], dtype=float)
    window = np.asarray([row["matrix_window_day"] for row in simulated], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    baseline_ratio = 2.4

    axes[0].plot(ratios, M_mat, "-o", color=COLORS["matrix"], label="$M_{mat}=A+H$")
    axes[0].plot(ratios, Q, "-s", color=COLORS["Q"], label="$Q$")
    axes[0].set_ylabel("Final retained mineral (mM P-eq.)")
    axes[0].legend()

    axes[1].plot(ratios, matrix_fraction, "-o", color=COLORS["H"])
    axes[1].set_ylabel("Final retained matrix fraction")
    axes[1].set_ylim(0.0, 1.0)

    axes[2].plot(ratios, t_A, "-o", color=COLORS["A"], label="$t_A$: $S_A>1$")
    axes[2].plot(ratios, t_Q, "-s", color=COLORS["Q"], label="$t_Q$: $S_Q>1$")
    finite_window = np.isfinite(window)
    if np.any(finite_window):
        axes[2].plot(ratios[finite_window], window[finite_window], "--", color="0.35", label="$t_Q-t_A$")
    axes[2].set_ylabel("Time (day)")
    axes[2].legend()

    for label, ax in zip("ABC", axes):
        ax.axvline(baseline_ratio, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_xlabel("$K_Q/K_A$")
        ax.grid(alpha=0.25)
        panel_label(ax, label)

    axes[0].set_title("Endpoint pathway allocation")
    axes[1].set_title("Retained matrix fraction")
    axes[2].set_title("Threshold timing")
    fig.tight_layout()
    save_figure(fig, output_dir, 11, formats)


GENERATORS = {
    2: figure_02, 3: figure_03, 4: figure_04, 5: figure_05,
    6: figure_06, 7: figure_07, 8: figure_08, 9: figure_09, 10: figure_10,
    11: figure_11,
}


def parse_figures(text: str) -> list[int]:
    requested: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            requested.update(range(start, end + 1))
        else:
            requested.add(int(part))
    invalid = requested.difference(GENERATORS)
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown figure number(s): {sorted(invalid)}")
    return sorted(requested)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate the revised competing-pathway manuscript figures.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--figures", type=parse_figures, default=list(range(2, 11)))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "figures")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--grid-size", type=int, default=40)
    parser.add_argument("--sobol-base", type=int, default=128)
    parser.add_argument("--jobs", type=int, default=1)
    add_model_arguments(parser)
    add_stochastic_arguments(parser)
    return parser


def generate_figures(
    figure_numbers: list[int],
    output_dir: Path,
    formats: tuple[str, ...],
    parameters: ModelParameters,
    settings: SimulationSettings,
    grid_size: int,
    sobol_base: int,
    jobs: int,
    seed: int,
    ensemble_runs: list[SimulationResult] | None = None,
) -> None:
    """Generate a selected, ordered set of manuscript figures."""
    configure_style()
    for number in figure_numbers:
        print(f"Generating {FIGURE_STEMS[number]} ...", flush=True)
        GENERATORS[number](output_dir=output_dir, formats=formats, parameters=parameters, settings=settings, grid_size=grid_size, sobol_base=sobol_base, jobs=jobs, seed=seed, ensemble_runs=ensemble_runs)


def main() -> None:
    args = build_parser().parse_args()
    if args.grid_size < 3 or args.sobol_base < 2 or args.jobs < 1:
        raise ValueError("grid-size >= 3, sobol-base >= 2, and jobs >= 1 are required")
    formats = tuple(value.strip().lower() for value in args.formats.split(",") if value.strip())
    output_dir = resolve_output_path(args.output_dir)
    stochastic = stochastic_from_args(args)
    parameters = parameters_from_args(args, stochastic)
    settings = settings_from_args(args)
    runs = simulate_ensemble(parameters, settings, stochastic)
    generate_figures(args.figures, output_dir, formats, parameters, settings, args.grid_size, args.sobol_base, args.jobs, args.seed, runs)
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
