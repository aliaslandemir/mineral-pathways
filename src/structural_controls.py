"""
Structural controls for the reduced ACP-to-apatite model.

These controls are model ablations, not alternative calibrated models. They use
the same baseline parameters, solver settings, and medium-renewal protocol as
the main model, then remove one structural assumption at a time.

python -m src.structural_controls --figure-formats pdf,png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PLOT_DEFAULTS, RUN_DEFAULTS, build_base_params
from src.io_utils import create_run_directory, parse_figure_formats, save_figure
from src.model import (
    MIN_DENOMINATOR,
    MIN_K_A,
    SimulationResult,
    matrix_support_factor,
    matrix_support_factor_at_time,
    synchronize_hydrolysis_parameters,
    transient_metrics,
    validate_simulation_state,
)
from src.visualization import clean_axis, configure_matplotlib


VARIANT_ORDER = ("full", "no_acp", "no_ascorbate", "no_threshold")
VARIANT_LABELS = {
    "full": "Full model",
    "no_acp": "No ACP compartment",
    "no_ascorbate": r"No ascorbate modulation",
    "no_threshold": "No threshold",
}
VARIANT_COLORS = {
    "full": "#1f4e79",
    "no_acp": "#d95f02",
    "no_ascorbate": "#2a9d8f",
    "no_threshold": "#7b3294",
}
ACP_COLOR = "#1f4e79"
HAP_COLOR = "#2a9d8f"


@dataclass(frozen=True)
class Fluxes:
    Rf: float
    Rm: float
    J_bgp: float
    S_A: float
    f_mat: float


def _project_state_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise RuntimeError("State trajectory contains nonfinite values.")
    min_value = float(np.min(arr))
    if min_value < -1e-3:
        raise RuntimeError(f"State trajectory contains negative values below tolerance: {min_value:.6g}")
    return np.maximum(arr, 0.0)


def _variant_fluxes(t: float, y: np.ndarray, params: dict[str, float], variant: str) -> Fluxes:
    C, P, B, A, H = [max(float(v), 0.0) for v in y]
    S_A = max((C ** params["alpha"] * P ** params["beta"]) / params["_K_A"], 0.0)
    capacity = max(0.0, 1.0 - (A + H) / params["_M_max"])
    f_mat = 1.0 if variant == "no_ascorbate" else matrix_support_factor_at_time(t, params)

    if variant == "no_threshold":
        formation_drive = max(S_A, 0.0) ** params["n"]
    else:
        formation_drive = max(S_A - 1.0, 0.0) ** params["n"] if S_A > 1.0 else 0.0

    Rf = params["k_f"] * formation_drive * capacity
    if variant == "no_acp":
        Rm = 0.0
    else:
        Rm = params["k_m"] * A * f_mat
    J_bgp = params["k_hyd"] * B

    if C <= 0.0:
        Rf = 0.0
        Rm = 0.0
    if P <= 0.0:
        Rf = 0.0
    if B <= 0.0:
        J_bgp = 0.0
    if A <= 0.0:
        Rm = 0.0
    return Fluxes(float(Rf), float(Rm), float(J_bgp), float(S_A), float(f_mat))


def _variant_rhs(t: float, y: np.ndarray, params: dict[str, float], variant: str) -> list[float]:
    C, P, B, A, H = [max(float(v), 0.0) for v in y]
    flux = _variant_fluxes(t, y, params, variant)
    if variant == "no_acp":
        dCdt = -params["r_H"] * flux.Rf
        dPdt = flux.J_bgp - flux.Rf
        dBdt = -flux.J_bgp
        dAdt = 0.0
        dHdt = flux.Rf
    else:
        extra_C_maturation = params["r_H"] - params["r_A"]
        dCdt = -params["r_A"] * flux.Rf - extra_C_maturation * flux.Rm
        dPdt = flux.J_bgp - flux.Rf
        dBdt = -flux.J_bgp
        dAdt = flux.Rf - flux.Rm
        dHdt = flux.Rm
    del C, P, B, A, H
    return [dCdt, dPdt, dBdt, dAdt, dHdt]


def simulate_variant(
    params: dict[str, float | str],
    *,
    variant: str,
    t_end: float,
    dt: float,
    media_interval: float,
) -> SimulationResult:
    if variant not in VARIANT_ORDER:
        raise ValueError(f"Unknown structural control variant: {variant}")

    p = dict(params)
    p["media_interval"] = float(media_interval)
    p["media_renewal_fraction"] = min(max(float(p.get("media_renewal_fraction", 1.0)), 0.0), 1.0)
    synchronize_hydrolysis_parameters(p, float(dt), float(media_interval))
    p["k_f"] = float(p["k_f"])
    p["k_m"] = float(p["k_m"])
    p["M_max"] = max(float(p["M_max"]), MIN_DENOMINATOR)
    p["_K_A"] = max(float(p["K_A"]), MIN_K_A)
    p["_M_max"] = p["M_max"]
    p["r_H"] = float(max(float(p["r_H"]), float(p["r_A"])))

    media_days = list(np.arange(media_interval, t_end, media_interval))
    breakpoints = media_days + [float(t_end)]
    y = np.array([p["C_ext"], p["P_ext"], p["beta_GP"], 0.0, 0.0], dtype=float)

    t0 = 0.0
    times_all: list[np.ndarray] = []
    states_all: list[np.ndarray] = []
    reset_status_all: list[np.ndarray] = []
    cycle_index_all: list[np.ndarray] = []
    cycle_start_all: list[np.ndarray] = []
    time_since_reset_all: list[np.ndarray] = []

    for cycle_idx, t1 in enumerate(breakpoints):
        p["_cycle_start"] = t0
        p["_cycle_length"] = t1 - t0
        n_steps = max(2, int(np.ceil((t1 - t0) / dt)) + 1)
        t_eval = np.linspace(t0, t1, n_steps)
        sol = solve_ivp(
            _variant_rhs,
            (t0, t1),
            y,
            args=(p, variant),
            t_eval=t_eval,
            method=str(p["solver_method"]),
            rtol=float(p["solver_rtol"]),
            atol=float(p["solver_atol"]),
            max_step=max(float(dt), MIN_DENOMINATOR),
        )
        if not sol.success:
            raise RuntimeError(sol.message)

        if times_all:
            t_segment = sol.t[1:]
            y_segment = sol.y[:, 1:]
        else:
            t_segment = sol.t
            y_segment = sol.y
        y_segment = _project_state_matrix(y_segment)
        status = np.full(t_segment.shape, "continuous", dtype=object)
        if np.isclose(t1, media_days).any() and status.size:
            status[-1] = "pre_reset"
        times_all.append(t_segment)
        states_all.append(y_segment)
        reset_status_all.append(status)
        cycle_index_all.append(np.full(t_segment.shape, cycle_idx, dtype=int))
        cycle_start_all.append(np.full(t_segment.shape, t0, dtype=float))
        time_since_reset_all.append(t_segment - t0)

        y = _project_state_matrix(sol.y[:, -1].reshape(5, 1))[:, 0].copy()
        if np.isclose(t1, media_days).any():
            renew = float(p["media_renewal_fraction"])
            y_reset = y.copy()
            y_reset[0] = (1.0 - renew) * y[0] + renew * max(float(p["C_ext"]), 0.0)
            y_reset[1] = (1.0 - renew) * y[1] + renew * max(float(p["P_ext"]), 0.0)
            y_reset[2] = (1.0 - renew) * y[2] + renew * max(float(p["beta_GP"]), 0.0)
            times_all.append(np.array([t1]))
            states_all.append(y_reset.reshape(5, 1))
            reset_status_all.append(np.array(["post_reset"], dtype=object))
            cycle_index_all.append(np.array([cycle_idx + 1], dtype=int))
            cycle_start_all.append(np.array([t1], dtype=float))
            time_since_reset_all.append(np.array([0.0], dtype=float))
            y = y_reset
        t0 = t1

    t = np.concatenate(times_all)
    Y = _project_state_matrix(np.concatenate(states_all, axis=1))
    C, P, B, A, H = Y
    reset_status = np.concatenate(reset_status_all)
    cycle_index = np.concatenate(cycle_index_all)
    cycle_start = np.concatenate(cycle_start_all)
    time_since_reset = np.concatenate(time_since_reset_all)

    fluxes = np.array([_variant_fluxes(ti, yi, p, variant) for ti, yi in zip(t, Y.T)], dtype=object)
    Rf = np.array([flux.Rf for flux in fluxes], dtype=float)
    Rm = np.array([flux.Rm for flux in fluxes], dtype=float)
    J_bgp = np.array([flux.J_bgp for flux in fluxes], dtype=float)
    S_A = np.array([flux.S_A for flux in fluxes], dtype=float)
    f_mat = np.array([flux.f_mat for flux in fluxes], dtype=float)

    sim = SimulationResult(
        t=t,
        C=C,
        P=P,
        B=B,
        A=A,
        H=H,
        J_bgp=J_bgp,
        Rf=Rf,
        Rm=Rm,
        S_A=S_A,
        f_MV=np.ones(len(t), dtype=float),
        f_mat=f_mat,
        media_days=media_days,
        reset_status=reset_status,
        cycle_index=cycle_index,
        cycle_start=cycle_start,
        time_since_reset=time_since_reset,
    )
    validate_simulation_state(sim)
    return sim


def first_positive_time(t: np.ndarray, values: np.ndarray, threshold: float = 1e-8) -> float:
    idx = np.flatnonzero(np.asarray(values, dtype=float) > float(threshold))
    return float(t[int(idx[0])]) if idx.size else float("nan")


def first_h_exceeds_a_time(sim: SimulationResult) -> float:
    total = sim.A + sim.H
    diff = sim.H - sim.A
    idx = np.flatnonzero((diff > 0.0) & (total > 1e-8))
    if idx.size == 0:
        return float("nan")
    j = int(idx[0])
    if j == 0:
        return float(sim.t[j])
    return float(np.interp(0.0, [diff[j - 1], diff[j]], [sim.t[j - 1], sim.t[j]]))


def variant_metrics(variant: str, sim: SimulationResult) -> dict[str, float | str | bool]:
    metrics = transient_metrics(sim)
    A_peak = float(metrics["A_peak"])
    H_final = float(metrics["H_final"])
    A_final = float(metrics["A_final"])
    formation_onset = first_positive_time(sim.t, sim.Rf)
    h_onset = first_positive_time(sim.t, sim.H, threshold=0.01 * max(H_final, 1e-8))
    h_exceeds_a = first_h_exceeds_a_time(sim)
    return {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "formation_onset_days": formation_onset,
        "H_1pct_onset_days": h_onset,
        "A_peak": A_peak,
        "A_peak_time_days": float(metrics["t_peak_days"]),
        "A_final": A_final,
        "H_final": H_final,
        "phi_H_final": float(metrics["phi_H_final"]),
        "A_decline_fraction": float(metrics["decline_fraction"]),
        "H_exceeds_A_time_days": h_exceeds_a,
        "has_delayed_onset": bool(np.isfinite(formation_onset) and formation_onset > 0.25),
        "has_transient_ACP_peak": bool(A_peak > 1e-6 and A_final < 0.75 * A_peak),
        "has_two_timescale_behavior": bool(np.isfinite(h_exceeds_a) and np.isfinite(formation_onset) and h_exceeds_a - formation_onset > 5.0),
    }


def simulate_all_controls(params: dict[str, float | str], t_end: float, dt: float, media_interval: float) -> dict[str, SimulationResult]:
    return {
        variant: simulate_variant(params, variant=variant, t_end=t_end, dt=dt, media_interval=media_interval)
        for variant in VARIANT_ORDER
    }


def plot_structural_trajectories(
    out_dir: Path,
    sims: dict[str, SimulationResult],
    figure_formats: tuple[str, ...],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0), sharex=True, sharey=True, constrained_layout=True)
    for ax, variant in zip(axes.ravel(), VARIANT_ORDER):
        sim = sims[variant]
        ax.plot(sim.t, sim.A, color=ACP_COLOR, linewidth=1.8, label="ACP-like A")
        ax.plot(sim.t, sim.H, color=HAP_COLOR, linewidth=1.8, label="Apatite-like H")
        ax.set_title(VARIANT_LABELS[variant])
        clean_axis(ax)
        ax.set_xlim(0.0, float(np.max(sim.t)))
        ax.set_ylim(0.0, 2.1)
        if variant == "full":
            ax.legend(frameon=False, loc="upper left")
    fig.supxlabel("Time (days)")
    fig.supylabel("Mineral pool (mM phosphate equivalents)")
    save_figure(fig, out_dir, "structural_control_trajectories", figure_formats, dpi=350)


def plot_renewal_response(
    out_dir: Path,
    renewal_df: pd.DataFrame,
    figure_formats: tuple[str, ...],
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.1), constrained_layout=True)
    for variant in VARIANT_ORDER:
        data = renewal_df[renewal_df["variant"] == variant].sort_values("media_interval_days")
        ax.plot(
            data["media_interval_days"],
            data["H_final"],
            marker="o",
            linewidth=1.7,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant],
        )
    clean_axis(ax)
    ax.set_xlabel("Medium-renewal interval (days)")
    ax.set_ylabel("Day-30 apatite-like H (mM phosphate equivalents)")
    ax.set_title("Renewal-Interval Response of Structural Controls")
    ax.set_xticks(sorted(renewal_df["media_interval_days"].unique()))
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, out_dir, "structural_control_renewal_response", figure_formats, dpi=350)


def run_structural_controls(
    *,
    t_end: float = RUN_DEFAULTS.t_end,
    dt: float = RUN_DEFAULTS.dt,
    media_interval: float = RUN_DEFAULTS.media_interval,
    figure_formats: tuple[str, ...] = ("pdf", "png"),
) -> Path:
    configure_matplotlib(PLOT_DEFAULTS)
    out_dir = create_run_directory("structural_controls", "controls")
    params = build_base_params()
    sims = simulate_all_controls(params, float(t_end), float(dt), float(media_interval))

    trajectory_rows = []
    for variant, sim in sims.items():
        for t, A, H, Rf, Rm, S_A, f_mat in zip(sim.t, sim.A, sim.H, sim.Rf, sim.Rm, sim.S_A, sim.f_mat):
            trajectory_rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "time_days": float(t),
                    "A": float(A),
                    "H": float(H),
                    "Rf": float(Rf),
                    "Rm": float(Rm),
                    "S_A": float(S_A),
                    "f_mat": float(f_mat),
                }
            )
    pd.DataFrame(trajectory_rows).to_csv(out_dir / "structural_control_trajectories.csv", index=False)

    metrics_df = pd.DataFrame([variant_metrics(variant, sim) for variant, sim in sims.items()])
    metrics_df.to_csv(out_dir / "structural_control_metrics.csv", index=False)

    renewal_rows = []
    for interval in (1.0, 2.0, 3.0):
        for variant in VARIANT_ORDER:
            sim = simulate_variant(params, variant=variant, t_end=float(t_end), dt=float(dt), media_interval=interval)
            row = variant_metrics(variant, sim)
            row["media_interval_days"] = interval
            renewal_rows.append(row)
    renewal_df = pd.DataFrame(renewal_rows)
    renewal_df.to_csv(out_dir / "structural_control_renewal_response.csv", index=False)

    behavior_df = metrics_df[
        [
            "variant",
            "variant_label",
            "has_delayed_onset",
            "has_transient_ACP_peak",
            "has_two_timescale_behavior",
            "formation_onset_days",
            "A_peak",
            "A_peak_time_days",
            "H_exceeds_A_time_days",
            "H_final",
        ]
    ].copy()
    behavior_df.to_csv(out_dir / "structural_control_behavior_matrix.csv", index=False)

    if figure_formats:
        plot_structural_trajectories(out_dir, sims, figure_formats)
        plot_renewal_response(out_dir, renewal_df, figure_formats)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run structural-control ablations for the reduced mineralization model.")
    parser.add_argument("--t-end", "--t_end", dest="t_end", type=float, default=RUN_DEFAULTS.t_end)
    parser.add_argument("--dt", type=float, default=RUN_DEFAULTS.dt)
    parser.add_argument("--media-interval", "--media_interval", dest="media_interval", type=float, default=RUN_DEFAULTS.media_interval)
    parser.add_argument("--figure-formats", default="pdf,png", help="Comma-separated: pdf,png; empty skips figures.")
    args = parser.parse_args()
    out_dir = run_structural_controls(
        t_end=args.t_end,
        dt=args.dt,
        media_interval=args.media_interval,
        figure_formats=parse_figure_formats(args.figure_formats),
    )
    print(f"Saved structural control outputs to: {out_dir}")
    print("These are model ablations, not alternative calibrated models.")


if __name__ == "__main__":
    main()
