
"""

Single-point-scaled external consistency comparison of total
calcium-rich model mineral with ARS experimental data

Lutrell2019 PloS One. DOI: 10.1371/journal.pone.0204197

python -m src.validation_ars \
    --n-runs 100 \
    --t-end 30 \
    --dt 0.05 \
    --media-interval 2 \
    --C-ext 1.8018 \
    --P-ext 0.906 \
    --beta-GP 10.0 \
    --eta-48h 0.22 \
    --K-A 18.0 \
    --k-f 1.0 \
    --M-max 2.0 \
    --k-m 0.2 \
    --a-AA0 0.28 \
    --t-AA-half-life 24.0 \
    --figure-formats pdf,png

"""



from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ARS experimental data
T_DAYS = np.array([4, 7, 10, 14, 21, 28], dtype=float)
VALIDATION_DAYS = np.array([4, 7, 10, 14, 21], dtype=float)
CALIBRATION_DAY = 28.0
ARS_MEAN = np.array([2.0, 4.0, 5.0, 12.0, 15.0, 30.0], dtype=float)
ARS_SE = np.array([2.0, 3.0, 3.0, 9.0, 8.0, 6.0], dtype=float)
PLOT_HILL_FITS = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DEFAULTS, PLOT_DEFAULTS, RUN_DEFAULTS, STOCHASTIC_DEFAULTS, build_base_params
from src.io_utils import create_run_directory, parse_figure_formats, save_figure
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
    parser.add_argument("--k-f", "--k_f", dest="k_f", type=float, default=MODEL_DEFAULTS.k_f)
    parser.add_argument("--M-max", "--M_max", dest="M_max", type=float, default=MODEL_DEFAULTS.M_max)
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
        "k_f",
        "M_max",
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


def hill_saturation(t: np.ndarray, h_max: float, t50: float, n: float) -> np.ndarray:
    t_arr = np.asarray(t, dtype=float)
    t_power = np.power(np.maximum(t_arr, 0.0), n)
    return h_max * t_power / (np.power(t50, n) + t_power)


def estimate_t50(days: np.ndarray, values: np.ndarray) -> float:
    half_max = 0.5 * float(np.max(values))
    if np.any(values >= half_max):
        idx = int(np.argmax(values >= half_max))
        if idx == 0:
            return float(days[idx])
        return float(np.interp(half_max, values[idx - 1 : idx + 1], days[idx - 1 : idx + 1]))
    return float(np.median(days))


def fit_hill_saturation(
    days: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray | None = None,
) -> tuple[tuple[float, float, float] | None, str]:
    days = np.asarray(days, dtype=float)
    values = np.asarray(values, dtype=float)
    sigma_fit = None if sigma is None else np.asarray(sigma, dtype=float)

    arrays = [days, values]
    if sigma_fit is not None:
        arrays.append(sigma_fit)
    if any(not np.all(np.isfinite(arr)) for arr in arrays):
        return None, "failed: nonfinite values"
    if len(days) < 3:
        return None, "failed: fewer than 3 points"
    if sigma_fit is not None and np.any(sigma_fit <= 0):
        return None, "failed: nonpositive sigma"

    h_guess = max(float(np.max(values)) * 1.15, 1e-6)
    t50_guess = max(estimate_t50(days, values), 1e-6)
    try:
        popt, _ = curve_fit(
            hill_saturation,
            days,
            values,
            p0=(h_guess, t50_guess, 3.0),
            sigma=sigma_fit,
            absolute_sigma=True,
            bounds=([1e-9, 1e-6, 0.1], [np.inf, float(np.max(days)) * 10.0, 10.0]),
            maxfev=20000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        return None, f"failed: {exc}"

    return tuple(float(v) for v in popt), "ok"


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


def require_finite_array(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name} contains nonfinite values.")


def require_days_available(days: np.ndarray, required_days: np.ndarray, context: str) -> None:
    missing = [day for day in required_days if not np.any(np.isclose(days, day))]
    if missing:
        missing_text = ", ".join(f"{day:g}" for day in missing)
        raise RuntimeError(f"{context} is missing required day(s): {missing_text}.")


def sample_at_days(sim_t: np.ndarray, values: np.ndarray, days: np.ndarray) -> np.ndarray:
    sim_t = np.asarray(sim_t, dtype=float)
    values = np.asarray(values, dtype=float)
    days = np.asarray(days, dtype=float)
    require_finite_array("simulation time", sim_t)
    require_finite_array("simulation values", values)
    require_finite_array("requested interpolation days", days)
    if len(sim_t) != len(values):
        raise RuntimeError("Simulation time and value arrays must have the same length.")
    if np.any(np.diff(sim_t) < 0):
        raise RuntimeError("Simulation time must be nondecreasing for interpolation.")
    if float(np.min(days)) < float(np.min(sim_t)) or float(np.max(days)) > float(np.max(sim_t)):
        raise RuntimeError("Requested interpolation day lies outside the simulation time range.")
    return np.interp(days, sim_t, values)


def experimental_dataframe() -> pd.DataFrame:
    require_days_available(T_DAYS, np.append(VALIDATION_DAYS, CALIBRATION_DAY), "ARS experimental data")
    require_finite_array("ARS mean", ARS_MEAN)
    require_finite_array("ARS SE", ARS_SE)
    if np.any(ARS_SE <= 0):
        raise RuntimeError("ARS SE must be positive.")
    return pd.DataFrame({"day": T_DAYS, "exp_mean": ARS_MEAN, "exp_se": ARS_SE})


def simulate_apatite_mineral_samples(base_params: dict[str, float], n_runs: int) -> pd.DataFrame:
    return simulate_apatite_mineral_samples_for_run(
        base_params=base_params,
        n_runs=n_runs,
        t_end=RUN_DEFAULTS.t_end,
        dt=RUN_DEFAULTS.dt,
        media_interval=RUN_DEFAULTS.media_interval,
    )


def simulate_apatite_mineral_samples_for_run(
    base_params: dict[str, float],
    n_runs: int,
    t_end: float,
    dt: float,
    media_interval: float,
) -> pd.DataFrame:
    rows = []
    samples = build_stochastic_samples(int(n_runs), seed=RUN_DEFAULTS.seed)

    for stochastic_idx, sample in enumerate(samples):
        pr = apply_stochastic_variability(base_params, sample)
        sim = simulate(
            pr,
            t_end=float(t_end),
            dt=float(dt),
            media_interval=float(media_interval),
        )
        acp_ca_days = sample_at_days(sim.t, sample["r_A"] * sim.A, T_DAYS)
        apatite_ca_days = sample_at_days(sim.t, sample["r_H"] * sim.H, T_DAYS)
        total_ca_days = acp_ca_days + apatite_ca_days
        require_finite_array("apatite-like modeled calcium mineral", apatite_ca_days)

        for day, a_val, h_val, total_val in zip(T_DAYS, acp_ca_days, apatite_ca_days, total_ca_days):
            rows.append(
                {
                    "stochastic_run": stochastic_idx + 1,
                    "day": float(day),
                    "r_A": float(sample["r_A"]),
                    "r_H": float(sample["r_H"]),
                    "acp_bound_Ca": float(a_val),
                    "apatite_bound_Ca": float(h_val),
                    "total_bound_Ca": float(total_val),
                }
                )

    model_df = pd.DataFrame(rows)
    required_columns = ["day", "acp_bound_Ca", "apatite_bound_Ca", "total_bound_Ca"]
    if model_df.empty:
        raise RuntimeError("No model validation samples were generated.")
    if model_df[required_columns].isna().any().any():
        raise RuntimeError("Model validation samples contain missing values.")
    require_finite_array("model validation samples", model_df[required_columns].to_numpy(dtype=float))
    require_days_available(model_df["day"].to_numpy(dtype=float), T_DAYS, "model validation samples")
    return model_df


def summarize_model_samples(model_df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        model_df.groupby("day", as_index=False)
        .agg(
            model_mean=("total_bound_Ca", "mean"),
            model_p2_5=("total_bound_Ca", lambda x: float(np.percentile(x, 2.5))),
            model_p97_5=("total_bound_Ca", lambda x: float(np.percentile(x, 97.5))),
            n=("total_bound_Ca", "size"),
        )
        .sort_values("day")
    )
    require_finite_array("model summary statistics", stats[["model_mean", "model_p2_5", "model_p97_5"]].to_numpy(dtype=float))
    require_days_available(stats["day"].to_numpy(dtype=float), T_DAYS, "model summary statistics")
    return stats


def fit_observation_scale_at_day28(merged: pd.DataFrame) -> tuple[float, float]:
    """Scale model output to ARS units using only the day-28 point."""
    calibration = merged.loc[np.isclose(merged["day"], CALIBRATION_DAY)]
    if len(calibration) != 1:
        raise RuntimeError("Exactly one calibration point is required.")

    model_value = float(calibration["model_mean"].iloc[0])
    experimental_value = float(calibration["exp_mean"].iloc[0])
    if model_value <= 0 or not np.isfinite(model_value):
        raise RuntimeError("Day-28 model value must be positive and finite.")
    if not np.isfinite(experimental_value):
        raise RuntimeError("Day-28 experimental value must be finite.")

    scale = experimental_value / model_value
    intercept = 0.0
    return float(scale), intercept


def add_scaled_model_columns(merged: pd.DataFrame, norm_scale: float, intercept: float) -> pd.DataFrame:
    out = merged.copy()
    for source, target in (
        ("model_mean", "model_mean_norm"),
        ("model_p2_5", "model_p2_5_norm"),
        ("model_p97_5", "model_p97_5_norm"),
    ):
        out[target] = out[source] * norm_scale + intercept
    out["norm_scale"] = norm_scale
    out["observation_intercept"] = intercept
    out["observation_model"] = "single-point scale at day 28"
    out["is_calibration_day"] = np.isclose(out["day"], CALIBRATION_DAY)
    out["is_validation_day"] = out["day"].isin(VALIDATION_DAYS)
    return out


def compute_pointwise_metrics(merged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    validation = merged.loc[merged["is_validation_day"]].copy()
    require_days_available(validation["day"].to_numpy(dtype=float), VALIDATION_DAYS, "pointwise validation data")

    metric_values = validation[["model_mean_norm", "exp_mean"]].to_numpy(dtype=float)
    require_finite_array("pointwise validation values", metric_values)
    residuals = validation["model_mean_norm"].to_numpy(dtype=float) - validation["exp_mean"].to_numpy(dtype=float)
    exp_values = validation["exp_mean"].to_numpy(dtype=float)
    model_values = validation["model_mean_norm"].to_numpy(dtype=float)

    exp_range = float(np.max(exp_values) - np.min(exp_values))
    if exp_range <= 0 or not np.isfinite(exp_range):
        raise RuntimeError("Experimental validation range must be positive for normalized RMSE.")
    if np.std(exp_values) <= 0 or np.std(model_values) <= 0:
        raise RuntimeError("Pearson r is undefined for constant validation values.")

    metrics = {
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "mae": float(np.mean(np.abs(residuals))),
        "nrmse": float(np.sqrt(np.mean(residuals**2)) / exp_range),
        "pearson_r": float(np.corrcoef(model_values, exp_values)[0, 1]),
    }
    for name, value in metrics.items():
        if not np.isfinite(value):
            raise RuntimeError(f"{name} is nonfinite.")

    out = merged.copy()
    for name, value in metrics.items():
        out[name] = value
    return out, metrics


def add_descriptive_hill_fits(
    merged: pd.DataFrame,
    run_hill_fits: bool = PLOT_HILL_FITS,
) -> tuple[pd.DataFrame, tuple[float, float, float] | None, tuple[float, float, float] | None]:
    out = merged.copy()
    if not run_hill_fits:
        out["model_hill_status"] = "not run"
        out["exp_hill_status"] = "not run"
        out["model_hill_fit"] = np.nan
        out["model_hill_Hmax"] = np.nan
        out["model_hill_t50"] = np.nan
        out["model_hill_n"] = np.nan
        out["exp_hill_fit"] = np.nan
        out["exp_hill_Hmax"] = np.nan
        out["exp_hill_t50"] = np.nan
        out["exp_hill_n"] = np.nan
        return out, None, None

    validation = out.loc[out["is_validation_day"]].copy()
    days = validation["day"].to_numpy(dtype=float)
    model_values = validation["model_mean_norm"].to_numpy(dtype=float)
    exp_values = validation["exp_mean"].to_numpy(dtype=float)
    exp_sigma = validation["exp_se"].to_numpy(dtype=float)

    model_params, model_status = fit_hill_saturation(days, model_values)
    exp_params, exp_status = fit_hill_saturation(days, exp_values, sigma=exp_sigma)

    out["model_hill_status"] = model_status
    out["exp_hill_status"] = exp_status
    if model_params is not None:
        out["model_hill_fit"] = hill_saturation(out["day"].to_numpy(dtype=float), *model_params)
        out["model_hill_Hmax"] = model_params[0]
        out["model_hill_t50"] = model_params[1]
        out["model_hill_n"] = model_params[2]
    else:
        out["model_hill_fit"] = np.nan
        out["model_hill_Hmax"] = np.nan
        out["model_hill_t50"] = np.nan
        out["model_hill_n"] = np.nan

    if exp_params is not None:
        out["exp_hill_fit"] = hill_saturation(out["day"].to_numpy(dtype=float), *exp_params)
        out["exp_hill_Hmax"] = exp_params[0]
        out["exp_hill_t50"] = exp_params[1]
        out["exp_hill_n"] = exp_params[2]
    else:
        out["exp_hill_fit"] = np.nan
        out["exp_hill_Hmax"] = np.nan
        out["exp_hill_t50"] = np.nan
        out["exp_hill_n"] = np.nan

    return out, model_params, exp_params


def plot_validation(
    out_dir: Path,
    merged: pd.DataFrame,
    metrics: dict[str, float],
    model_hill_params: tuple[float, float, float] | None,
    exp_hill_params: tuple[float, float, float] | None,
    figure_formats: tuple[str, ...],
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.1), constrained_layout=True)

    ax.fill_between(
        merged["day"].to_numpy(dtype=float),
        merged["model_p2_5_norm"].to_numpy(dtype=float),
        merged["model_p97_5_norm"].to_numpy(dtype=float),
        color="#4292c6",
        alpha=0.30,
        linewidth=0.8,
        edgecolor="#08519c",
        label="Variability induced by sampled effective Ca/P ratios",
    )
    ax.plot(
        merged["day"],
        merged["model_mean_norm"],
        color="#08519c",
        linewidth=1.9,
        label="Model (day-28 scaled)",
    )
    ax.scatter(
        merged["day"],
        merged["model_mean_norm"],
        color="#08519c",
        marker="o",
        s=24,
        label="_nolegend_",
        zorder=3,
    )
    if model_hill_params is not None:
        t_fit = np.linspace(float(np.min(T_DAYS)), float(np.max(T_DAYS)), 300)
        ax.plot(
            t_fit,
            hill_saturation(t_fit, *model_hill_params),
            color="#08519c",
            linewidth=1.7,
            label="Model Hill",
        )
    if exp_hill_params is not None:
        t_fit = np.linspace(float(np.min(T_DAYS)), float(np.max(T_DAYS)), 300)
        ax.plot(
            t_fit,
            hill_saturation(t_fit, *exp_hill_params),
            color="#cb181d",
            linewidth=1.5,
            linestyle="--",
            label="Experimental Hill",
        )
    ax.errorbar(
        merged["day"],
        merged["exp_mean"],
        yerr=merged["exp_se"],
        fmt="s",
        color="#cb181d",
        ecolor="#cb181d",
        elinewidth=1.1,
        capsize=3,
        markersize=4.5,
        label="Experiment ± SE",
    )

    clean_axis(ax)
    ax.set_xlabel("Culture time (days)")
    ax.set_ylabel("Mineralization signal (a.u.)")
    ax.set_title("ARS and day-28-scaled total calcium-rich mineral")
    ax.set_xticks(T_DAYS)
    ax.set_xlim(float(np.min(T_DAYS)) - 0.5, float(np.max(T_DAYS)) + 1)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.98,
        0.04,
        f"RMSE(4-21 d) = {metrics['rmse']:.2f} a.u.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
        color="#222222",
    )

    save_figure(
        fig,
        out_dir,
        "ars_consistency_experimental_vs_model_day28_scaled",
        figure_formats,
        dpi=350,
    )


def run_validation(
    n_runs: int = RUN_DEFAULTS.n_runs,
    figure_formats: tuple[str, ...] = ("pdf", "png"),
    base_params: dict[str, float] | None = None,
    t_end: float = RUN_DEFAULTS.t_end,
    dt: float = RUN_DEFAULTS.dt,
    media_interval: float = RUN_DEFAULTS.media_interval,
) -> Path:
    configure_matplotlib(PLOT_DEFAULTS)
    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.grid": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
        }
    )

    out_dir = create_run_directory("validation", "ars")

    exp_df = experimental_dataframe()
    base_params = build_base_params() if base_params is None else dict(base_params)
    model_df = simulate_apatite_mineral_samples_for_run(
        base_params,
        n_runs=n_runs,
        t_end=float(t_end),
        dt=float(dt),
        media_interval=float(media_interval),
    )
    model_df.to_csv(out_dir / "total_mineral_day_samples.csv", index=False)

    model_stats = summarize_model_samples(model_df)
    merged = model_stats.merge(exp_df, on="day", how="inner")
    require_days_available(merged["day"].to_numpy(dtype=float), T_DAYS, "merged ARS validation data")

    norm_scale, intercept = fit_observation_scale_at_day28(merged)
    merged = add_scaled_model_columns(merged, norm_scale, intercept)
    merged, metrics = compute_pointwise_metrics(merged)
    merged, model_hill_params, exp_hill_params = add_descriptive_hill_fits(merged)
    merged.to_csv(out_dir / "ars_validation_summary.csv", index=False)

    plot_validation(out_dir, merged, metrics, model_hill_params, exp_hill_params, figure_formats)

    metrics_df = pd.DataFrame(
        [
            {
                "validation_days": ",".join(f"{day:g}" for day in VALIDATION_DAYS),
                "calibration_day": CALIBRATION_DAY,
                "observation_model": "single-point scale at day 28",
                "norm_scale": norm_scale,
                "observation_intercept": intercept,
                **metrics,
            }
        ]
    )
    metrics_df.to_csv(out_dir / "ars_validation_metrics.csv", index=False)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare total calcium-rich mineral against ARS measurements after day-28 scaling."
    )
    parser.add_argument("--n-runs", type=int, default=RUN_DEFAULTS.n_runs)
    parser.add_argument("--t-end", "--t_end", dest="t_end", type=float, default=RUN_DEFAULTS.t_end)
    parser.add_argument("--dt", type=float, default=RUN_DEFAULTS.dt)
    parser.add_argument("--media-interval", "--media_interval", dest="media_interval", type=float, default=RUN_DEFAULTS.media_interval)
    parser.add_argument("--figure-formats", default="pdf,png", help="Comma-separated: pdf,png; empty skips figures.")
    add_model_override_arguments(parser)
    args = parser.parse_args()
    out_dir = run_validation(
        args.n_runs,
        parse_figure_formats(args.figure_formats),
        base_params=apply_model_overrides(build_base_params(), args),
        t_end=args.t_end,
        dt=args.dt,
        media_interval=args.media_interval,
    )
    print(f"Saved ARS validation outputs to: {out_dir}")


if __name__ == "__main__":
    main()
