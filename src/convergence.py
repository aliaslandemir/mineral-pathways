"""Baseline numerical step-size check; run with: python convergence.py."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mineral-pathways-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

from src.analysis import first_crossing, inventory_error
import src.model as model_module
from src.model import MineralizationModel, ModelParameters, SimulationSettings


STEPS = (0.10, 0.05, 0.025, 0.0125)
PLOT_INTERVAL = 0.05  # Output interpolation only; never passed to the ODE or limiter.
OUTPUT = Path(__file__).resolve().parent / "output" / "convergence"
# The nine Sobol ranges in src/figures.py (figure_07) and paper/sup.tex.
SOBOL_RANGES = {
    "B_ext": (5.0, 20.0),
    "eta_48h": (0.10, 0.50),
    "K_A": (24.0, 48.0),
    "K_Q": (50.0, 132.0),
    "k_f": (0.05, 1.00),
    "M_max": (0.25, 4.0),
    "k_m": (0.03, 3.0),
    "k_b": (0.03, 3.0),
    "Q_max": (0.01, 4.0),
}


class CheckedModel(MineralizationModel):
    """Record the actual clipping interval and whether the limiter acts."""

    def __init__(self, parameters: ModelParameters):
        super().__init__(parameters)
        self.clipping_intervals: set[float] = set()
        self.limited_rhs_calls = 0
        self.rhs_calls = 0

    def rhs(self, t: float, state: np.ndarray, cycle_start: float, availability_dt: float) -> np.ndarray:
        self.clipping_intervals.add(float(availability_dt))
        self.rhs_calls += 1
        nominal = self.rates(state, None)
        limited = self.rates(state, availability_dt)
        if any(limited[i] < nominal[i] for i in (1, 2, 3)):
            self.limited_rhs_calls += 1
        return super().rhs(t, state, cycle_start, availability_dt)


def relative_percent(value: float, reference: float) -> float:
    return 100.0 * (value - reference) / reference if reference != 0.0 else float("nan")


def isolated_step_run(parameters: ModelParameters, max_step: float, clipping_dt: float) -> tuple[float, float, float]:
    """Change one solver control while preserving the 0.05-day output grid."""
    original = model_module.solve_ivp

    def adjusted_solver(*args, **kwargs):
        kwargs["max_step"] = max_step
        cycle_start, _ = kwargs["args"]
        kwargs["args"] = (cycle_start, clipping_dt)
        return original(*args, **kwargs)

    settings = SimulationSettings(t_end=28.0, dt=0.05, media_interval=2.0,
                                  method="RK45", rtol=1e-6, atol=1e-9)
    with patch.object(model_module, "solve_ivp", adjusted_solver):
        model = CheckedModel(parameters)
        result = model.simulate(settings)
    if model.clipping_intervals != {clipping_dt}:
        raise RuntimeError("Diagnostic clipping interval was not applied")
    return float(result.M_mat[-1]), float(result.Q[-1]), float(result.M_tot[-1])


def stress_test(base: ModelParameters) -> str:
    """Screen 50 Sobol-range combinations and refine active-limiter cases."""
    names = list(SOBOL_RANGES)
    bounds = np.asarray([SOBOL_RANGES[name] for name in names], dtype=float)
    unit_samples = qmc.LatinHypercube(d=len(names), seed=42).random(n=50)
    samples = qmc.scale(unit_samples, bounds[:, 0], bounds[:, 1])
    settings_by_step = {
        step: SimulationSettings(t_end=28.0, dt=step, media_interval=2.0,
                                 method="RK45", rtol=1e-6, atol=1e-9)
        for step in (0.05, 0.025, 0.0125)
    }
    rows: list[dict[str, object]] = []
    parameters_by_case: dict[int, ModelParameters] = {}

    def run(case_id: int, parameters: ModelParameters, step: float, stage: str) -> dict[str, object]:
        model = CheckedModel(parameters)
        result = model.simulate(settings_by_step[step])
        if model.clipping_intervals != {step}:
            raise RuntimeError(f"Case {case_id}: clipping interval {step} was not applied")
        return {
            "case_id": case_id, "stage": stage, "step_day": step,
            **{name: getattr(parameters, name) for name in names},
            "limiter_activated": model.limited_rhs_calls > 0,
            "limited_rhs_calls": model.limited_rhs_calls,
            "rhs_calls": model.rhs_calls,
            "matrix_mineral_day28": float(result.M_mat[-1]),
            "solution_mineral_day28": float(result.Q[-1]),
            "total_mineral_day28": float(result.M_tot[-1]),
            "matrix_fraction_day28": result.final_matrix_fraction,
        }

    for case_id, sample in enumerate(samples, start=1):
        parameters = base.changed(**dict(zip(names, sample)))
        parameters_by_case[case_id] = parameters
        rows.append(run(case_id, parameters, 0.05, "screen"))

    active_rows = sorted((row for row in rows if row["limiter_activated"]),
                         key=lambda row: (-row["limited_rhs_calls"], row["case_id"]))
    selected_ids = [row["case_id"] for row in active_rows[:3]]
    metric_names = ("matrix_mineral_day28", "solution_mineral_day28",
                    "total_mineral_day28", "matrix_fraction_day28")
    comparisons: list[tuple[float, float, bool]] = []
    for case_id in selected_ids:
        group = [next(row for row in rows if row["case_id"] == case_id)]
        for step in (0.025, 0.0125):
            group.append(run(case_id, parameters_by_case[case_id], step, "refinement"))
        finest = group[-1]
        for row in group:
            for name in metric_names:
                row[name + "_difference_from_finest"] = row[name] - finest[name]
        rows.extend(group[1:])
        coarse_change = max(abs(group[0][name] - group[1][name]) for name in metric_names)
        fine_change = max(abs(group[1][name] - group[2][name]) for name in metric_names)
        largest_from_finest = max(abs(row[name] - finest[name]) for row in group for name in metric_names)
        comparisons.append((largest_from_finest, fine_change, fine_change <= coarse_change))

    fields = ("case_id", "stage", "step_day", *names, "limiter_activated",
              "limited_rhs_calls", "rhs_calls", *metric_names,
              *(name + "_difference_from_finest" for name in metric_names))
    with (OUTPUT / "stress_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if not active_rows:
        return ("Stress test: 50 fixed-seed Latin-hypercube combinations across all nine "
                "manuscript Sobol ranges produced no limiter activations at 0.05 day. "
                "No refinement cases were constructed.")
    activity = ", ".join(f"case {row['case_id']} ({row['limited_rhs_calls']} calls)"
                         for row in active_rows[:3])
    largest = max(item[0] for item in comparisons)
    stable = sum(item[2] for item in comparisons)
    return (f"Stress test: {len(active_rows)} of 50 fixed-seed Latin-hypercube cases "
            f"activated the limiter at 0.05 day. Highest-activity cases: {activity}. "
            f"Across their 0.05, 0.025, and 0.0125-day runs, the largest absolute "
            f"difference from the finest result among the three mineral endpoints "
            f"and matrix fraction was {largest:.4g} (endpoint units or fraction units, "
            f"respectively). The maximum absolute adjacent-step change decreased "
            f"from 0.05→0.025 to 0.025→0.0125 day in {stable} of "
            f"{len(selected_ids)} selected cases.")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    parameters = ModelParameters()
    plot_time = np.arange(0.0, 28.0 + PLOT_INTERVAL / 2, PLOT_INTERVAL)
    rows: list[dict[str, float | int]] = []
    curves: list[np.ndarray] = []

    for step in STEPS:
        # The current model uses settings.dt for max_step, t_eval, and availability_dt.
        settings = SimulationSettings(t_end=28.0, dt=step, media_interval=2.0,
                                      method="RK45", rtol=1e-6, atol=1e-9)
        model = CheckedModel(parameters)
        result = model.simulate(settings)
        if model.clipping_intervals != {step}:
            raise RuntimeError(f"Expected clipping interval {step}, got {model.clipping_intervals}")
        rows.append({
            "step_day": step,
            "max_step_day": step,
            "clipping_dt_day": step,
            "plot_interval_day": PLOT_INTERVAL,
            "matrix_mineral_day28": float(result.M_mat[-1]),
            "solution_mineral_day28": float(result.Q[-1]),
            "total_mineral_day28": float(result.M_tot[-1]),
            "matrix_fraction_day28": result.final_matrix_fraction,
            "SA_first_crossing_day": first_crossing(result, 1.0, "S_A"),
            "SQ_first_crossing_day": first_crossing(result, 1.0, "S_Q"),
            "max_calcium_balance_residual": float(np.max(inventory_error(result, "calcium_inventory"))),
            "max_phosphate_balance_residual": float(np.max(inventory_error(result, "phosphate_inventory"))),
            "limited_rhs_calls": model.limited_rhs_calls,
            "rhs_calls": model.rhs_calls,
        })
        curves.append(result.values_at(plot_time, "M_tot"))

    finest = rows[-1]
    endpoints = ("matrix_mineral_day28", "solution_mineral_day28", "total_mineral_day28")
    for row in rows:
        for name in endpoints:
            row[name + "_relative_difference_percent"] = relative_percent(row[name], finest[name])
        row["matrix_fraction_difference_pp"] = 100.0 * (
            row["matrix_fraction_day28"] - finest["matrix_fraction_day28"]
        )
        for name in ("SA_first_crossing_day", "SQ_first_crossing_day"):
            row[name + "_difference_day"] = row[name] - finest[name]

    with (OUTPUT / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.3), constrained_layout=True)
    colors = ("#1b4f72", "#2980b9", "#b35900", "#8e44ad")
    styles = ("-", "--", "-.", ":")
    for step, curve, color, style in zip(STEPS, curves, colors, styles):
        axes[0].plot(plot_time, curve, color=color, linestyle=style, linewidth=1.8,
                     label=f"{step:g} d")
    axes[0].set(xlim=(0, 28), xlabel="Time (days)", ylabel="Total mineral (mM P-eq.)")
    axes[0].legend(title="Numerical step", frameon=False, loc="lower right", fontsize=8)
    axes[0].text(-0.14, 1.03, "A", transform=axes[0].transAxes, fontweight="bold", fontsize=11)

    axes[1].plot(STEPS, [row["matrix_fraction_difference_pp"] for row in rows],
                 color="#1b4f72", marker="o", markersize=5, linewidth=1.5)
    axes[1].axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    axes[1].set(xlabel="Numerical step (days)",
                ylabel="Difference in day-28 matrix fraction\n(percentage points)",
                xticks=STEPS, xlim=(0.0, 0.11))
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.5f}"))
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].text(-0.14, 1.03, "B", transform=axes[1].transAxes, fontweight="bold", fontsize=11)
    fig.savefig(OUTPUT / "convergence.pdf")
    fig.savefig(OUTPUT / "convergence.png", dpi=300)
    plt.close(fig)

    max_endpoint_change = max(abs(row[name + "_relative_difference_percent"])
                              for row in rows for name in endpoints)
    max_fraction_change = max(abs(row["matrix_fraction_difference_pp"]) for row in rows)
    max_crossing_change = max(abs(row[name + "_difference_day"]) for row in rows
                              for name in ("SA_first_crossing_day", "SQ_first_crossing_day"))
    max_ca = max(row["max_calcium_balance_residual"] for row in rows)
    max_p = max(row["max_phosphate_balance_residual"] for row in rows)
    active = sum(row["limited_rhs_calls"] for row in rows)
    summary = (
        "Numerical step-size check (baseline, 28 days). The RK45 maximum step and "
        "ion-availability clipping interval were both set to 0.10, 0.05, 0.025, "
        "and 0.0125 day. Parameters, initial conditions, 2-day medium renewal, "
        "and solver tolerances (rtol=1e-6, atol=1e-9) were held fixed. "
        "RK45 internal steps were adaptive; the 0.05-day plot grid was used only "
        "to interpolate completed trajectories. Relative to the finest run, "
        f"day-28 mineral endpoints differed by at most {max_endpoint_change:.4g}%, "
        f"matrix fraction by {max_fraction_change:.4g} percentage points, "
        f"and interpolated threshold times by {max_crossing_change:.4g} day. "
        f"Maximum within-cycle relative calcium and phosphate inventory residuals "
        f"were {max_ca:.3g} and {max_p:.3g}, respectively. "
        f"The ion-availability limiter acted on {active} of "
        f"{sum(row['rhs_calls'] for row in rows)} RHS evaluations. "
    )
    if active == 0:
        summary += "This baseline test does not assess step sensitivity when clipping is active."
    else:
        summary += "The limiter was active, so its step dependence is included in this comparison."
    if max_endpoint_change > 0.0:
        baseline = rows[1]
        reference = tuple(baseline[name] for name in endpoints)
        max_only = isolated_step_run(parameters, max_step=0.10, clipping_dt=0.05)
        clip_only = isolated_step_run(parameters, max_step=0.05, clipping_dt=0.10)
        max_only_change = max(abs(relative_percent(value, ref)) for value, ref in zip(max_only, reference))
        clip_only_change = max(abs(relative_percent(value, ref)) for value, ref in zip(clip_only, reference))
        summary += (
            " In a one-control diagnostic relative to the 0.05-day baseline, "
            f"changing only max_step to 0.10 day changed mineral endpoints by at most "
            f"{max_only_change:.4g}%; changing only clipping dt to 0.10 day changed "
            f"them by at most {clip_only_change:.4g}%."
        )
    stress_summary = stress_test(parameters)
    (OUTPUT / "summary.txt").write_text(summary + "\n\n" + stress_summary + "\n", encoding="utf-8")
    print(summary)
    print(stress_summary)
    print(f"Saved CSV, PDF, PNG, and summary in {OUTPUT}")


if __name__ == "__main__":
    main()
