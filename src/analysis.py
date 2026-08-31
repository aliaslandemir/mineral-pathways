"""Reusable analyses for the competing-pathway model.

Run the detailed analysis from the repository root::

    python -m src.analysis \
    --t-end 30 \
    --dt 0.05 \
    --media-interval 2 \
    --K_Q 66.0 \
    --k_b 0.2 \
    --Q_max 1.0 \
    --n_b 1.0 \
    --output-dir output/analysis
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import numpy as np

from src.model import (
    DEFAULT_OUTPUT_DIR,
    EPS,
    FIXED_MODEL_ASSUMPTIONS,
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


T = TypeVar("T")
U = TypeVar("U")
KQ_KA_RATIOS = (1.1, 1.25, 1.5, 1.75, 2.0, 2.4, 3.0, 4.0)


def ordered_map(function: Callable[[T], U], items: Iterable[T], jobs: int = 1) -> list[U]:
    values = list(items)
    if jobs <= 1:
        return [function(value) for value in values]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(function, values))


def integrate_by_cycle(values: np.ndarray, result: SimulationResult) -> float:
    total = 0.0
    for cycle in np.unique(result.cycle):
        mask = (result.cycle == cycle) & (result.reset_status != "post_reset")
        if np.count_nonzero(mask) > 1:
            total += float(np.trapezoid(np.asarray(values)[mask], result.t[mask]))
    return total


def first_crossing(result: SimulationResult, level: float, field: str = "S_A") -> float:
    values = np.asarray(getattr(result, field), dtype=float)
    for cycle in np.unique(result.cycle):
        indices = np.where((result.cycle == cycle) & (result.reset_status != "post_reset"))[0]
        above = np.where(values[indices] >= level)[0]
        if not above.size:
            continue
        j = int(above[0])
        if j == 0:
            return float(result.t[indices[j]])
        left, right = indices[j - 1], indices[j]
        weight = (level - values[left]) / (values[right] - values[left])
        return float(result.t[left] + weight * (result.t[right] - result.t[left]))
    return np.nan


def output_metrics(result: SimulationResult) -> dict[str, float]:
    peak_index = int(np.argmax(result.A))
    integrated_formation = integrate_by_cycle(result.R_f, result)
    integrated_bulk = integrate_by_cycle(result.R_b, result)
    peak_a = float(result.A[peak_index])
    return {
        "terminal H": float(result.H[-1]),
        "terminal M_mat": float(result.M_mat[-1]),
        "terminal Q": float(result.Q[-1]),
        "terminal M_tot": float(result.M_tot[-1]),
        "peak A": peak_a,
        "threshold t_A": MineralizationModel(result.parameters).threshold_times()[0],
        "time of peak A": float(result.t[peak_index]),
        "ACP persistence": integrate_by_cycle(result.A, result) / max(peak_a, EPS),
        "matrix fraction": integrated_formation / max(integrated_formation + integrated_bulk, EPS),
    }


def detailed_metrics(result: SimulationResult) -> dict[str, float | str | int]:
    """Return endpoint, timing, exposure, pathway, and balance diagnostics."""
    p = result.parameters
    matrix_time = integrate_by_cycle((result.S_A > 1.0).astype(float), result)
    bulk_time = integrate_by_cycle((result.S_Q > 1.0).astype(float), result)
    formation = integrate_by_cycle(result.R_f, result)
    maturation = integrate_by_cycle(result.R_m, result)
    bulk = integrate_by_cycle(result.R_b, result)
    t_a, t_q = MineralizationModel(p).threshold_times()
    metrics: dict[str, float | str | int] = {
        "n_recorded_points": int(result.t.size),
        "n_renewals": int(np.count_nonzero(result.reset_status == "post_reset")),
        "analytical_t_A_day": t_a,
        "analytical_t_Q_day": t_q,
        "analytical_matrix_window_day": t_q - t_a if np.isfinite(t_a) and np.isfinite(t_q) else np.nan,
        "numerical_first_SA_1_day": first_crossing(result, 1.0),
        "numerical_first_SQ_1_day": first_crossing(result, 1.0, "S_Q"),
        "solution_K_Q": p.K_Q,
        "K_Q_over_K_A": p.K_Q / p.K_A,
        "maximum_S_A": float(np.max(result.S_A)),
        "matrix_accessible_time_day": matrix_time,
        "bulk_accessible_time_day": bulk_time,
        "matrix_accessible_fraction": matrix_time / result.settings.t_end,
        "bulk_accessible_fraction": bulk_time / result.settings.t_end,
        "integrated_R_f_mM": formation,
        "integrated_R_m_mM": maturation,
        "integrated_R_b_mM": bulk,
        "cumulative_matrix_formation_fraction": formation / max(formation + bulk, EPS),
        "maturation_fraction_of_matrix_formation": maturation / max(formation, EPS),
        "final_B_mM": float(result.B[-1]),
        "final_C_mM": float(result.C[-1]),
        "final_P_mM": float(result.P[-1]),
        "final_A_mM_P_eq": float(result.A[-1]),
        "final_H_mM_P_eq": float(result.H[-1]),
        "final_Q_mM_P_eq": float(result.Q[-1]),
        "final_M_mat_mM_P_eq": float(result.M_mat[-1]),
        "final_M_tot_mM_P_eq": float(result.M_tot[-1]),
        "final_retained_matrix_fraction": result.final_matrix_fraction,
        "peak_A_mM_P_eq": float(np.max(result.A)),
        "time_peak_A_day": float(result.t[int(np.argmax(result.A))]),
        "peak_R_f_mM_day": float(np.max(result.R_f)),
        "peak_R_m_mM_day": float(np.max(result.R_m)),
        "peak_R_b_mM_day": float(np.max(result.R_b)),
        "max_relative_phosphate_balance_error": float(np.max(inventory_error(result, "phosphate_inventory"))),
        "max_relative_calcium_balance_error": float(np.max(inventory_error(result, "calcium_inventory"))),
    }
    return metrics


def kq_ka_ratio_sweep_rows(parameters: ModelParameters, settings: SimulationSettings) -> list[dict[str, float | str]]:
    """Run the existing model across structural K_Q/K_A ratios."""
    rows: list[dict[str, float | str]] = []
    K_A = parameters.K_A
    for ratio in KQ_KA_RATIOS:
        K_Q = ratio * K_A
        row: dict[str, float | str] = {
            "KQ_KA_ratio": ratio,
            "K_A": K_A,
            "K_Q": K_Q,
            "t_A_day": np.nan,
            "t_Q_day": np.nan,
            "matrix_window_day": np.nan,
            "integrated_R_f": np.nan,
            "integrated_R_b": np.nan,
            "final_M_mat": np.nan,
            "final_Q": np.nan,
            "final_M_tot": np.nan,
            "final_matrix_fraction": np.nan,
            "status": "simulated",
        }
        try:
            result = MineralizationModel(parameters.changed(K_A=K_A, K_Q=K_Q)).simulate(settings)
        except ValueError as exc:
            row["status"] = str(exc)
            rows.append(row)
            continue

        t_A = first_crossing(result, 1.0, "S_A")
        t_Q = first_crossing(result, 1.0, "S_Q")
        row.update(
            {
                "t_A_day": t_A,
                "t_Q_day": t_Q,
                "matrix_window_day": t_Q - t_A if np.isfinite(t_A) and np.isfinite(t_Q) else np.nan,
                "integrated_R_f": integrate_by_cycle(result.R_f, result),
                "integrated_R_b": integrate_by_cycle(result.R_b, result),
                "final_M_mat": float(result.M_mat[-1]),
                "final_Q": float(result.Q[-1]),
                "final_M_tot": float(result.M_tot[-1]),
                "final_matrix_fraction": result.final_matrix_fraction,
            }
        )
        rows.append(row)
    return rows


def save_kq_ka_ratio_sweep(parameters: ModelParameters, settings: SimulationSettings, output_dir: Path) -> list[dict[str, float | str]]:
    """Save the structural sensitivity sweep for K_Q/K_A."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = kq_ka_ratio_sweep_rows(parameters, settings)
    fieldnames = [
        "KQ_KA_ratio",
        "K_A",
        "K_Q",
        "t_A_day",
        "t_Q_day",
        "matrix_window_day",
        "integrated_R_f",
        "integrated_R_b",
        "final_M_mat",
        "final_Q",
        "final_M_tot",
        "final_matrix_fraction",
        "status",
    ]
    with (output_dir / "KQ_KA_ratio_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def save_kh_ka_ratio_sweep(parameters: ModelParameters, settings: SimulationSettings, output_dir: Path) -> list[dict[str, float | str]]:
    """Backward-compatible wrapper for the K_Q/K_A sweep."""
    return save_kq_ka_ratio_sweep(parameters, settings, output_dir)


def cycle_metrics(result: SimulationResult) -> list[dict[str, float | int]]:
    """Summarize soluble states, retained changes, rates, and thresholds per renewal cycle."""
    rows: list[dict[str, float | int]] = []
    for cycle in np.unique(result.cycle):
        indices = np.where((result.cycle == cycle) & (result.reset_status != "pre_reset"))[0]
        pre_indices = np.where((result.cycle == cycle) & (result.reset_status != "post_reset"))[0]
        if not indices.size or not pre_indices.size:
            continue
        combined = np.unique(np.concatenate((indices, pre_indices)))
        first, last = int(combined[0]), int(combined[-1])
        rows.append(
            {
                "cycle": int(cycle),
                "start_day": float(result.t[first]),
                "end_day": float(result.t[last]),
                "B_start": float(result.B[first]), "B_end": float(result.B[last]),
                "C_start": float(result.C[first]), "C_end": float(result.C[last]),
                "P_start": float(result.P[first]), "P_end": float(result.P[last]),
                "M_mat_start": float(result.M_mat[first]), "M_mat_end": float(result.M_mat[last]),
                "Q_start": float(result.Q[first]), "Q_end": float(result.Q[last]),
                "maximum_S_A": float(np.max(result.S_A[combined])),
                "integrated_R_f": float(np.trapezoid(result.R_f[combined], result.t[combined])),
                "integrated_R_m": float(np.trapezoid(result.R_m[combined], result.t[combined])),
                "integrated_R_b": float(np.trapezoid(result.R_b[combined], result.t[combined])),
            }
        )
    return rows


def _json_value(value: object) -> object:
    if isinstance(value, (np.floating, float)) and not np.isfinite(float(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_analysis(result: SimulationResult, output_dir: Path) -> dict[str, float | str | int]:
    """Write all detailed analysis products and return the summary metrics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = detailed_metrics(result)
    with (output_dir / "summary_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows((name, value) for name, value in metrics.items())
    (output_dir / "summary_metrics.json").write_text(
        json.dumps({name: _json_value(value) for name, value in metrics.items()}, indent=2) + "\n", encoding="utf-8"
    )
    rows = cycle_metrics(result)
    with (output_dir / "cycle_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    configuration = {
        "model_parameters": asdict(result.parameters),
        "fixed_model_assumptions": FIXED_MODEL_ASSUMPTIONS,
        "simulation_settings": asdict(result.settings),
    }
    (output_dir / "run_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "conservation_timeseries.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_day", "cycle", "phosphate_inventory", "calcium_inventory", "phosphate_relative_error", "calcium_relative_error"])
        writer.writerows(zip(result.t, result.cycle, result.phosphate_inventory, result.calcium_inventory, inventory_error(result, "phosphate_inventory"), inventory_error(result, "calcium_inventory")))
    return metrics


def save_ensemble_analysis(runs: list[SimulationResult], output_dir: Path) -> None:
    """Save per-realization metrics and ensemble uncertainty summaries."""
    metric_rows = [detailed_metrics(run) for run in runs]
    numeric_names = [name for name, value in metric_rows[0].items() if isinstance(value, (int, float, np.number))]
    with (output_dir / "stochastic_run_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "r_A", "r_H", *metric_rows[0]])
        writer.writeheader()
        for index, (run, row) in enumerate(zip(runs, metric_rows)):
            writer.writerow({"run": index, "r_A": run.parameters.r_A, "r_H": run.parameters.r_H, **row})
    with (output_dir / "stochastic_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "sd", "q025", "median", "q975"])
        for name in numeric_names:
            values = np.asarray([row[name] for row in metric_rows], dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                writer.writerow([name, np.mean(finite), np.std(finite), *np.quantile(finite, [0.025, 0.5, 0.975])])


def sample_ca_p_parameters(seed: int, count: int, base: ModelParameters | None = None) -> list[ModelParameters]:
    rng = np.random.default_rng(seed)
    base = base or ModelParameters()
    return [
        base.changed(r_A=1.20 + 0.29 * xi_a, r_H=1.50 + 0.17 * xi_h)
        for xi_a, xi_h in rng.random((count, 2))
    ]


def inventory_error(result: SimulationResult, inventory: str) -> np.ndarray:
    values = np.asarray(getattr(result, inventory), dtype=float)
    relative = np.zeros_like(values)
    for cycle in np.unique(result.cycle):
        mask = result.cycle == cycle
        reference = values[np.where(mask)[0][0]]
        relative[mask] = np.abs(values[mask] - reference) / max(abs(reference), EPS)
    return relative


def evaluate_parameter_vector(task: tuple[dict[str, float], SimulationSettings]) -> dict[str, float]:
    parameters, settings = task
    return output_metrics(MineralizationModel(ModelParameters(**parameters)).simulate(settings))


def parameter_dict(parameters: ModelParameters) -> dict[str, float]:
    return {name: float(value) for name, value in asdict(parameters).items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and save detailed competing-pathway analyses.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_model_arguments(parser)
    add_stochastic_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "analysis")
    args = parser.parse_args()
    stochastic = stochastic_from_args(args)
    result = MineralizationModel(parameters_from_args(args, stochastic)).simulate(settings_from_args(args))
    output_dir = resolve_output_path(args.output_dir)
    metrics = save_analysis(result, output_dir)
    save_kq_ka_ratio_sweep(result.parameters, result.settings, output_dir)
    save_ensemble_analysis(simulate_ensemble(result.parameters, result.settings, stochastic), output_dir)
    for name, value in metrics.items():
        print(f"{name}: {value:.6g}")
    print(f"Saved detailed analyses to {output_dir}")


if __name__ == "__main__":
    main()
