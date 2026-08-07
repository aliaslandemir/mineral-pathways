"""
ARS trajectory calibration of k_f and M_max.

This workflow optimizes only k_f and M_max. For each parameter pair it simulates
mean[r_A*A(t) + r_H*H(t)] using the same fixed stochastic Ca/P samples, then
estimates a single zero-intercept ARS observation scale analytically from all
six ARS points. This avoids day-28 normalization while keeping the optimizer
focused on the two model parameters.

python -m src.calibration_ars \
    --n-runs 100 \
    --t-end 30 \
    --dt 0.05 \
    --media-interval 2 \
    --C-ext 1.8018 \
    --P-ext 0.906 \
    --beta-GP 5.0 \
    --eta-48h 0.22 \
    --K-A 18.0 \
    --k-m 0.2 \
    --a-AA0 0.28 \
    --t-AA-half-life 24.0 \
    --n-starts 30 \
    --grid-size 60 \
    --jobs 6 \
    --figure-formats pdf,png
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os
import sys
from typing import Iterable
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

T_DAYS = np.array([4, 7, 10, 14, 21, 28], dtype=float)
ARS_MEAN = np.array([2.0, 4.0, 5.0, 12.0, 15.0, 30.0], dtype=float)
ARS_SE = np.array([2.0, 3.0, 3.0, 9.0, 8.0, 6.0], dtype=float)
KF_BOUNDS = (0.05, 3.0)
M_MAX_BOUNDS = (0.5, 10.0)
BOUND_REL_TOL = 0.01
RECOMMENDED_MIN_STARTS = 30
RECOMMENDED_MIN_GRID_SIZE = 60

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DEFAULTS, PLOT_DEFAULTS, RUN_DEFAULTS, STOCHASTIC_DEFAULTS, build_base_params
from src.io_utils import create_run_directory, parse_figure_formats, save_figure_to_directory
from src.model import simulate
from src.sampling import apply_stochastic_variability, build_ca_p_samples
from src.visualization import clean_axis, configure_matplotlib


def add_model_override_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--C-ext", "--C_ext", dest="C_ext", type=float, default=MODEL_DEFAULTS.C_ext)
    parser.add_argument("--P-ext", "--P_ext", dest="P_ext", type=float, default=MODEL_DEFAULTS.P_ext)
    parser.add_argument("--beta-GP", "--beta_GP", dest="beta_GP", type=float, default=MODEL_DEFAULTS.beta_GP)
    parser.add_argument("--eta-48h", "--eta_48h", dest="eta_48h", type=float, default=MODEL_DEFAULTS.eta_48h)
    parser.add_argument("--media-renewal-fraction", "--media_renewal_fraction", dest="media_renewal_fraction", type=float, default=MODEL_DEFAULTS.media_renewal_fraction)
    parser.add_argument("--alpha", type=float, default=MODEL_DEFAULTS.alpha)
    parser.add_argument("--beta", type=float, default=MODEL_DEFAULTS.beta)
    parser.add_argument("--n", type=float, default=MODEL_DEFAULTS.n)
    parser.add_argument("--K-A", "--K_A", dest="K_A", type=float, default=MODEL_DEFAULTS.K_A)
    parser.add_argument("--k-m", "--k_m", dest="k_m", type=float, default=MODEL_DEFAULTS.k_m)
    parser.add_argument("--solver-method", "--solver_method", dest="solver_method", type=str, default=RUN_DEFAULTS.solver_method)
    parser.add_argument("--solver-rtol", "--solver_rtol", dest="solver_rtol", type=float, default=RUN_DEFAULTS.solver_rtol)
    parser.add_argument("--solver-atol", "--solver_atol", dest="solver_atol", type=float, default=RUN_DEFAULTS.solver_atol)
    parser.add_argument("--a-AA0", "--a_AA0", dest="a_AA0", type=float, default=MODEL_DEFAULTS.a_AA0)
    parser.add_argument("--t-AA-half-life", "--t_AA_half_life", dest="t_AA_half_life", type=float, default=MODEL_DEFAULTS.t_AA_half_life)


def apply_model_overrides(base_params: dict[str, float], args: argparse.Namespace) -> dict[str, float]:
    params = dict(base_params)
    for name in (
        "C_ext",
        "P_ext",
        "beta_GP",
        "eta_48h",
        "media_renewal_fraction",
        "alpha",
        "beta",
        "n",
        "K_A",
        "k_m",
        "solver_method",
        "solver_rtol",
        "solver_atol",
        "a_AA0",
        "t_AA_half_life",
    ):
        value = getattr(args, name, None)
        if value is not None:
            params[name] = str(value) if name == "solver_method" else float(value)
    return params


def require_finite_array(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name} contains nonfinite values.")


def sample_at_days(sim_t: np.ndarray, values: np.ndarray, days: np.ndarray) -> np.ndarray:
    sim_t = np.asarray(sim_t, dtype=float)
    values = np.asarray(values, dtype=float)
    days = np.asarray(days, dtype=float)
    require_finite_array("simulation time", sim_t)
    require_finite_array("simulation values", values)
    if len(sim_t) != len(values):
        raise RuntimeError("Simulation time and value arrays must have the same length.")
    if np.any(np.diff(sim_t) < 0):
        raise RuntimeError("Simulation time must be nondecreasing for interpolation.")
    if float(np.min(days)) < float(np.min(sim_t)) or float(np.max(days)) > float(np.max(sim_t)):
        raise RuntimeError("Requested interpolation day lies outside the simulation time range.")
    return np.interp(days, sim_t, values)


def experimental_dataframe() -> pd.DataFrame:
    require_finite_array("ARS mean", ARS_MEAN)
    require_finite_array("ARS SE", ARS_SE)
    if np.any(ARS_SE <= 0):
        raise RuntimeError("ARS SE must be positive.")
    return pd.DataFrame({"day": T_DAYS, "ars_mean": ARS_MEAN, "ars_se": ARS_SE})


def build_stochastic_samples(n_runs: int, seed: int) -> list[dict[str, float]]:
    return build_ca_p_samples(
        n_runs=n_runs,
        seed=seed,
        r_A_min=STOCHASTIC_DEFAULTS.r_A_min,
        r_A_max=STOCHASTIC_DEFAULTS.r_A_max,
        r_H_min=STOCHASTIC_DEFAULTS.r_H_min,
        r_H_max=STOCHASTIC_DEFAULTS.r_H_max,
        xi_scale=STOCHASTIC_DEFAULTS.Xi,
    )


def simulate_total_model_at_days(
    *,
    base_params: dict[str, float],
    samples: list[dict[str, float]],
    k_f: float,
    M_max: float,
    t_end: float,
    dt: float,
    media_interval: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = dict(base_params)
    params["k_f"] = float(k_f)
    params["M_max"] = float(M_max)
    run_values = []

    for sample in samples:
        pr = apply_stochastic_variability(params, sample)
        sim = simulate(pr, t_end=float(t_end), dt=float(dt), media_interval=float(media_interval))
        total = sample["r_A"] * sim.A + sample["r_H"] * sim.H
        run_values.append(sample_at_days(sim.t, total, T_DAYS))

    values = np.asarray(run_values, dtype=float)
    require_finite_array("modeled total mineral samples", values)
    mean = np.mean(values, axis=0)
    if np.any(mean <= 0.0):
        raise RuntimeError("Modeled total mineral must be positive at all ARS days for scale calibration.")
    return mean, np.percentile(values, 2.5, axis=0), np.percentile(values, 97.5, axis=0)


def fit_zero_intercept_scale(model_values: np.ndarray) -> float:
    model_values = np.asarray(model_values, dtype=float)
    weights = 1.0 / np.square(ARS_SE)
    denom = float(np.sum(weights * model_values * model_values))
    if denom <= 0.0:
        raise RuntimeError("Cannot estimate ARS scale from nonpositive model denominator.")
    scale = float(np.sum(weights * model_values * ARS_MEAN) / denom)
    return max(scale, 0.0)


def scaled_metrics(model_values: np.ndarray, scale: float) -> dict[str, np.ndarray | float]:
    scaled = float(scale) * np.asarray(model_values, dtype=float)
    residual = (scaled - ARS_MEAN) / ARS_SE
    raw_residual = scaled - ARS_MEAN
    return {
        "ars_scale": float(scale),
        "scaled_model": scaled,
        "raw_residual": raw_residual,
        "weighted_residual": residual,
        "weighted_sse": float(np.sum(residual**2)),
        "weighted_rmse": float(np.sqrt(np.mean(residual**2))),
        "rmse_ars_units": float(np.sqrt(np.mean(raw_residual**2))),
        "relative_rmse": float(np.sqrt(np.mean(raw_residual**2)) / np.mean(ARS_MEAN)),
    }


class CalibrationEvaluator:
    def __init__(
        self,
        *,
        base_params: dict[str, float],
        samples: list[dict[str, float]],
        t_end: float,
        dt: float,
        media_interval: float,
    ) -> None:
        self.base_params = dict(base_params)
        self.samples = samples
        self.t_end = float(t_end)
        self.dt = float(dt)
        self.media_interval = float(media_interval)
        self.cache: dict[tuple[float, float], dict[str, np.ndarray | float]] = {}

    def evaluate(self, x: Iterable[float]) -> dict[str, np.ndarray | float]:
        k_f, M_max = [float(v) for v in x]
        key = (round(k_f, 12), round(M_max, 12))
        if key not in self.cache:
            mean, p2_5, p97_5 = simulate_total_model_at_days(
                base_params=self.base_params,
                samples=self.samples,
                k_f=k_f,
                M_max=M_max,
                t_end=self.t_end,
                dt=self.dt,
                media_interval=self.media_interval,
            )
            scale = fit_zero_intercept_scale(mean)
            metrics = scaled_metrics(mean, scale)
            self.cache[key] = {
                "k_f": k_f,
                "M_max": M_max,
                "model_total_mean": mean,
                "model_total_p2_5": p2_5,
                "model_total_p97_5": p97_5,
                "scaled_model_p2_5": scale * p2_5,
                "scaled_model_p97_5": scale * p97_5,
                "model_day28_normalized": mean / mean[-1],
                "experimental_day28_normalized": ARS_MEAN / ARS_MEAN[-1],
                **metrics,
            }
        return self.cache[key]

    def residuals(self, x: Iterable[float]) -> np.ndarray:
        return np.asarray(self.evaluate(x)["weighted_residual"], dtype=float)


def initial_points(n_starts: int, seed: int) -> np.ndarray:
    if int(n_starts) < RECOMMENDED_MIN_STARTS:
        warnings.warn(
            f"n_starts={n_starts} is exploratory; use at least {RECOMMENDED_MIN_STARTS} for final calibration.",
            RuntimeWarning,
            stacklevel=2,
        )
    rng = np.random.default_rng(int(seed))
    lower = np.array([KF_BOUNDS[0], M_MAX_BOUNDS[0]], dtype=float)
    upper = np.array([KF_BOUNDS[1], M_MAX_BOUNDS[1]], dtype=float)
    points = rng.uniform(lower, upper, size=(max(int(n_starts), 1), 2))
    points[0] = np.array([MODEL_DEFAULTS.k_f, MODEL_DEFAULTS.M_max], dtype=float)
    if len(points) > 1:
        points[1] = lower
    if len(points) > 2:
        points[2] = upper
    return points


def run_multistart_optimization(
    evaluator: CalibrationEvaluator,
    *,
    n_starts: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    best: dict[str, object] | None = None
    bounds = ([KF_BOUNDS[0], M_MAX_BOUNDS[0]], [KF_BOUNDS[1], M_MAX_BOUNDS[1]])

    for idx, start in enumerate(initial_points(n_starts, seed), start=1):
        result = least_squares(
            evaluator.residuals,
            x0=start,
            bounds=bounds,
            method="trf",
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
            max_nfev=200,
        )
        eval_result = evaluator.evaluate(result.x)
        row = {
            "start_index": idx,
            "start_k_f": float(start[0]),
            "start_M_max": float(start[1]),
            "k_f": float(result.x[0]),
            "M_max": float(result.x[1]),
            "ars_scale": float(eval_result["ars_scale"]),
            "weighted_sse": float(eval_result["weighted_sse"]),
            "weighted_rmse": float(eval_result["weighted_rmse"]),
            "rmse_ars_units": float(eval_result["rmse_ars_units"]),
            "relative_rmse": float(eval_result["relative_rmse"]),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(result.nfev),
            "cost": float(result.cost),
        }
        rows.append(row)
        if best is None or row["weighted_sse"] < float(best["weighted_sse"]):
            best = {**row, "optimizer_result": result}

    if best is None:
        raise RuntimeError("No optimizer starts were evaluated.")
    return pd.DataFrame(rows), best


def _evaluate_grid_task(task: tuple[dict[str, float], list[dict[str, float]], float, float, float, float, float]) -> dict[str, float]:
    base_params, samples, t_end, dt, media_interval, k_f, M_max = task
    mean, _, _ = simulate_total_model_at_days(
        base_params=base_params,
        samples=samples,
        k_f=k_f,
        M_max=M_max,
        t_end=t_end,
        dt=dt,
        media_interval=media_interval,
    )
    scale = fit_zero_intercept_scale(mean)
    metrics = scaled_metrics(mean, scale)
    return {
        "k_f": float(k_f),
        "M_max": float(M_max),
        "ars_scale": float(metrics["ars_scale"]),
        "weighted_sse": float(metrics["weighted_sse"]),
        "weighted_rmse": float(metrics["weighted_rmse"]),
        "rmse_ars_units": float(metrics["rmse_ars_units"]),
        "relative_rmse": float(metrics["relative_rmse"]),
    }


def dense_grid_scan(
    evaluator: CalibrationEvaluator,
    *,
    grid_size: int,
    jobs: int = 1,
) -> pd.DataFrame:
    if int(grid_size) < RECOMMENDED_MIN_GRID_SIZE:
        warnings.warn(
            f"grid_size={grid_size} is exploratory; use at least {RECOMMENDED_MIN_GRID_SIZE} for final error-surface assessment.",
            RuntimeWarning,
            stacklevel=2,
        )
    if int(jobs) < 1:
        raise ValueError("jobs must be at least 1.")

    kf_values = np.linspace(KF_BOUNDS[0], KF_BOUNDS[1], max(int(grid_size), 2))
    mmax_values = np.linspace(M_MAX_BOUNDS[0], M_MAX_BOUNDS[1], max(int(grid_size), 2))
    tasks = [(float(k_f), float(M_max)) for k_f in kf_values for M_max in mmax_values]

    if int(jobs) > 1 and len(tasks) > 1:
        worker_tasks = [
            (
                evaluator.base_params,
                evaluator.samples,
                evaluator.t_end,
                evaluator.dt,
                evaluator.media_interval,
                k_f,
                M_max,
            )
            for k_f, M_max in tasks
        ]
        chunksize = max(1, len(worker_tasks) // (int(jobs) * 8))
        with ProcessPoolExecutor(max_workers=int(jobs)) as executor:
            return pd.DataFrame(executor.map(_evaluate_grid_task, worker_tasks, chunksize=chunksize))

    return pd.DataFrame([_evaluate_grid_task((evaluator.base_params, evaluator.samples, evaluator.t_end, evaluator.dt, evaluator.media_interval, k_f, M_max)) for k_f, M_max in tasks])


def near_bound(value: float, bounds: tuple[float, float]) -> bool:
    width = float(bounds[1] - bounds[0])
    tol = BOUND_REL_TOL * width
    return bool(abs(float(value) - bounds[0]) <= tol or abs(float(value) - bounds[1]) <= tol)


def predictions_dataframe(evaluator: CalibrationEvaluator, best_k_f: float, best_M_max: float) -> pd.DataFrame:
    exp_df = experimental_dataframe()
    result = evaluator.evaluate((best_k_f, best_M_max))
    out = exp_df.copy()
    out["model_total_mean"] = np.asarray(result["model_total_mean"], dtype=float)
    out["model_total_p2_5"] = np.asarray(result["model_total_p2_5"], dtype=float)
    out["model_total_p97_5"] = np.asarray(result["model_total_p97_5"], dtype=float)
    out["ars_scale"] = float(result["ars_scale"])
    out["scaled_model"] = np.asarray(result["scaled_model"], dtype=float)
    out["scaled_model_p2_5"] = np.asarray(result["scaled_model_p2_5"], dtype=float)
    out["scaled_model_p97_5"] = np.asarray(result["scaled_model_p97_5"], dtype=float)
    out["raw_residual"] = np.asarray(result["raw_residual"], dtype=float)
    out["weighted_residual"] = np.asarray(result["weighted_residual"], dtype=float)
    out["experimental_day28_normalized"] = np.asarray(result["experimental_day28_normalized"], dtype=float)
    out["model_day28_normalized"] = np.asarray(result["model_day28_normalized"], dtype=float)
    return out


def plot_scale_calibration(
    out_dir: Path,
    predictions: pd.DataFrame,
    best: dict[str, object],
    figure_formats: tuple[str, ...],
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.1), constrained_layout=True)
    days = predictions["day"].to_numpy(dtype=float)
    ax.fill_between(
        days,
        predictions["scaled_model_p2_5"].to_numpy(dtype=float),
        predictions["scaled_model_p97_5"].to_numpy(dtype=float),
        color="#6baed6",
        alpha=0.28,
        linewidth=0.8,
        edgecolor="#2171b5",
        label="Sampled Ca/P variability",
    )
    ax.plot(days, predictions["scaled_model"], color="#08519c", linewidth=1.9, label="Model with fitted ARS scale")
    ax.scatter(days, predictions["scaled_model"], color="#08519c", s=24, zorder=3)
    ax.errorbar(
        days,
        predictions["ars_mean"],
        yerr=predictions["ars_se"],
        fmt="s",
        color="#cb181d",
        ecolor="#cb181d",
        elinewidth=1.1,
        capsize=3,
        markersize=4.5,
        label="ARS mean +/- SE",
    )
    clean_axis(ax)
    ax.set_xlabel("Culture time (days)")
    ax.set_ylabel("ARS signal (a.u.)")
    ax.set_title("ARS Scale-Calibrated Trajectory")
    ax.set_xticks(T_DAYS)
    ax.set_xlim(float(np.min(T_DAYS)) - 0.5, float(np.max(T_DAYS)) + 1)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.05,
        f"k_f={float(best['k_f']):.4g}, M_max={float(best['M_max']):.4g}\n"
        f"scale={float(best['ars_scale']):.4g}, SSE={float(best['weighted_sse']):.3g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        color="#222222",
    )
    save_figure_to_directory(fig, out_dir, "ars_scale_calibration", figure_formats, dpi=350)


def plot_error_surface(
    out_dir: Path,
    grid_df: pd.DataFrame,
    best: dict[str, object],
    figure_formats: tuple[str, ...],
) -> None:
    kf_values = np.sort(grid_df["k_f"].unique())
    mmax_values = np.sort(grid_df["M_max"].unique())
    surface = (
        grid_df.pivot(index="M_max", columns="k_f", values="weighted_sse")
        .reindex(index=mmax_values, columns=kf_values)
        .to_numpy(dtype=float)
    )
    fig, ax = plt.subplots(figsize=(6.3, 4.9), constrained_layout=True)
    mesh = ax.pcolormesh(kf_values, mmax_values, surface, shading="auto", cmap="viridis")
    contour_levels = np.linspace(float(np.nanmin(surface)), float(np.nanmax(surface)), 10)
    if np.all(np.isfinite(contour_levels)) and contour_levels[-1] > contour_levels[0]:
        ax.contour(kf_values, mmax_values, surface, levels=contour_levels, colors="white", linewidths=0.45, alpha=0.7)
    ax.scatter([float(best["k_f"])], [float(best["M_max"])], marker="x", s=72, linewidths=2.0, color="#d7191c", label="Optimizer optimum", zorder=4)
    ax.set_xlabel("k_f")
    ax.set_ylabel("M_max")
    ax.set_title("k_f-M_max Weighted SSE Surface")
    ax.set_xlim(*KF_BOUNDS)
    ax.set_ylim(*M_MAX_BOUNDS)
    ax.legend(frameon=False, loc="upper right")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Weighted SSE")
    clean_axis(ax)
    save_figure_to_directory(fig, out_dir, "kf_Mmax_error_surface", figure_formats, dpi=350)


def run_calibration(
    *,
    n_runs: int = RUN_DEFAULTS.n_runs,
    figure_formats: tuple[str, ...] = ("pdf", "png"),
    base_params: dict[str, float] | None = None,
    t_end: float = RUN_DEFAULTS.t_end,
    dt: float = RUN_DEFAULTS.dt,
    media_interval: float = RUN_DEFAULTS.media_interval,
    n_starts: int = RECOMMENDED_MIN_STARTS,
    grid_size: int = RECOMMENDED_MIN_GRID_SIZE,
    seed: int = RUN_DEFAULTS.seed,
    jobs: int = RUN_DEFAULTS.jobs,
) -> tuple[Path, dict[str, object]]:
    configure_matplotlib(PLOT_DEFAULTS)
    out_dir = create_run_directory("calibration", "ars_scaled")

    base_params = build_base_params() if base_params is None else dict(base_params)
    samples = build_stochastic_samples(int(n_runs), seed=int(seed))
    evaluator = CalibrationEvaluator(
        base_params=base_params,
        samples=samples,
        t_end=float(t_end),
        dt=float(dt),
        media_interval=float(media_interval),
    )

    all_starts, best = run_multistart_optimization(evaluator, n_starts=n_starts, seed=seed)
    all_starts.to_csv(out_dir / "calibration_all_starts.csv", index=False)

    grid_df = dense_grid_scan(evaluator, grid_size=grid_size, jobs=jobs)
    grid_df.to_csv(out_dir / "calibration_error_surface.csv", index=False)

    predictions = predictions_dataframe(evaluator, float(best["k_f"]), float(best["M_max"]))
    predictions.to_csv(out_dir / "calibration_predictions.csv", index=False)

    kf_near = near_bound(float(best["k_f"]), KF_BOUNDS)
    mmax_near = near_bound(float(best["M_max"]), M_MAX_BOUNDS)
    meets_original_sampling_spec = int(n_starts) >= RECOMMENDED_MIN_STARTS and int(grid_size) >= RECOMMENDED_MIN_GRID_SIZE
    params_df = pd.DataFrame(
        [
            {
                "calibration": "ARS scale-calibrated trajectory calibration",
                "objective": "weighted raw ARS residuals with analytically profiled zero-intercept scale",
                "k_f": float(best["k_f"]),
                "M_max": float(best["M_max"]),
                "ars_scale": float(best["ars_scale"]),
                "weighted_sse": float(best["weighted_sse"]),
                "weighted_rmse": float(best["weighted_rmse"]),
                "rmse_ars_units": float(best["rmse_ars_units"]),
                "relative_rmse": float(best["relative_rmse"]),
                "optimizer_success": bool(best["success"]),
                "optimizer_status": int(best["status"]),
                "optimizer_message": str(best["message"]),
                "k_f_near_bound": kf_near,
                "M_max_near_bound": mmax_near,
                "any_optimum_near_bound": bool(kf_near or mmax_near),
                "k_f_lower_bound": KF_BOUNDS[0],
                "k_f_upper_bound": KF_BOUNDS[1],
                "M_max_lower_bound": M_MAX_BOUNDS[0],
                "M_max_upper_bound": M_MAX_BOUNDS[1],
                "n_starts": int(n_starts),
                "grid_size": int(grid_size),
                "jobs": int(jobs),
                "n_runs": int(n_runs),
                "random_seed": int(seed),
                "meets_original_sampling_spec": bool(meets_original_sampling_spec),
                "warning": "The fitted ARS scale maps modeled mineral to assay signal; independent absolute mineral capacity still requires independent absolute mineral data.",
            }
        ]
    )
    params_df.to_csv(out_dir / "calibration_parameters.csv", index=False)

    if figure_formats:
        plot_scale_calibration(out_dir, predictions, best, figure_formats)
        plot_error_surface(out_dir, grid_df, best, figure_formats)

    best["k_f_near_bound"] = kf_near
    best["M_max_near_bound"] = mmax_near
    best["any_optimum_near_bound"] = bool(kf_near or mmax_near)
    best["meets_original_sampling_spec"] = bool(meets_original_sampling_spec)
    return out_dir, best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ARS scale-calibrated trajectory calibration of k_f and M_max against Luttrell ARS data."
    )
    parser.add_argument("--n-runs", type=int, default=RUN_DEFAULTS.n_runs)
    parser.add_argument("--t-end", "--t_end", dest="t_end", type=float, default=RUN_DEFAULTS.t_end)
    parser.add_argument("--dt", type=float, default=RUN_DEFAULTS.dt)
    parser.add_argument("--media-interval", "--media_interval", dest="media_interval", type=float, default=RUN_DEFAULTS.media_interval)
    parser.add_argument("--n-starts", type=int, default=RECOMMENDED_MIN_STARTS)
    parser.add_argument("--grid-size", type=int, default=RECOMMENDED_MIN_GRID_SIZE)
    parser.add_argument("--jobs", type=int, default=RUN_DEFAULTS.jobs, help="Worker processes for the dense grid scan.")
    parser.add_argument("--random-seed", "--seed", dest="seed", type=int, default=RUN_DEFAULTS.seed)
    parser.add_argument("--figure-formats", default="pdf,png", help="Comma-separated: pdf,png; empty skips figures.")
    add_model_override_arguments(parser)
    args = parser.parse_args()

    out_dir, best = run_calibration(
        n_runs=args.n_runs,
        figure_formats=parse_figure_formats(args.figure_formats),
        base_params=apply_model_overrides(build_base_params(), args),
        t_end=args.t_end,
        dt=args.dt,
        media_interval=args.media_interval,
        n_starts=args.n_starts,
        grid_size=args.grid_size,
        seed=args.seed,
        jobs=args.jobs,
    )
    print(f"Saved ARS scale-calibrated trajectory outputs to: {out_dir}")
    print("This objective does not normalize to day 28; it fits one zero-intercept ARS scale from all six points.")
    print("WARNING: The fitted ARS scale maps modeled mineral to assay signal; independent absolute mineral capacity still requires independent absolute mineral data.")
    print(f"best k_f: {float(best['k_f']):.8g}")
    print(f"best M_max: {float(best['M_max']):.8g}")
    print(f"ARS scale: {float(best['ars_scale']):.8g}")
    print(f"weighted SSE: {float(best['weighted_sse']):.8g}")
    print(f"weighted RMSE: {float(best['weighted_rmse']):.8g}")
    print(f"RMSE in ARS units: {float(best['rmse_ars_units']):.8g}")
    print(f"relative RMSE: {float(best['relative_rmse']):.8g}")
    print(f"optimizer success: {bool(best['success'])}")
    print(f"k_f near bound: {bool(best['k_f_near_bound'])}")
    print(f"M_max near bound: {bool(best['M_max_near_bound'])}")
    print(f"either optimum near a parameter bound: {bool(best['any_optimum_near_bound'])}")
    print(f"meets original start/grid sampling spec: {bool(best['meets_original_sampling_spec'])}")


if __name__ == "__main__":
    main()
