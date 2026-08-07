"""Archived phosphate-accounting figure export."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import save_figure

def phosphate_cycle_accounting(
    sim: Any,
    params: dict[str, float],
    media_interval: float,
) -> dict[str, np.ndarray | float]:
    t = np.asarray(sim.t, dtype=float)
    free_pi = np.asarray(sim.P, dtype=float)
    beta_gp_remaining = np.asarray(sim.B, dtype=float)
    release_rate = np.asarray(sim.J_bgp, dtype=float)
    formation_rate = np.asarray(sim.Rf, dtype=float)
    cycle_index = np.asarray(sim.cycle_index, dtype=int)
    cycle_start = np.asarray(sim.cycle_start, dtype=float)
    reset_status = np.asarray(sim.reset_status, dtype=object)

    beta_gp_cycle = np.zeros_like(t, dtype=float)
    removed_cycle = np.zeros_like(t, dtype=float)
    cycle_free_p0 = np.zeros_like(t, dtype=float)

    for cycle in np.unique(cycle_index):
        idx = np.where(cycle_index == cycle)[0]
        idx = idx[reset_status[idx] != "pre_reset"]
        if idx.size == 0:
            continue
        cycle_b0 = float(beta_gp_remaining[idx[0]])
        cycle_p0 = float(free_pi[idx[0]])
        for local_pos, i in enumerate(idx):
            cycle_free_p0[i] = cycle_p0
            beta_gp_cycle[i] = max(0.0, cycle_b0 - float(beta_gp_remaining[i]))
            if local_pos == 0:
                removed_cycle[i] = 0.0
                continue
            prev = int(idx[local_pos - 1])
            dt = float(t[i] - t[prev])
            if dt < -1e-10:
                raise ValueError("Simulation time vector is not monotonically increasing within a medium cycle.")
            removed_cycle[i] = removed_cycle[prev] + 0.5 * (formation_rate[prev] + formation_rate[i]) * max(dt, 0.0)

    balance_estimate = cycle_free_p0 + beta_gp_cycle - removed_cycle
    balance_error = free_pi - balance_estimate
    active = reset_status != "pre_reset"
    max_abs_error = float(np.max(np.abs(balance_error[active]))) if np.any(active) else 0.0

    return {
        "free_phosphate_mM": free_pi,
        "cycle_start_free_phosphate_mM": cycle_free_p0,
        "betaGP_remaining_mM": beta_gp_remaining,
        "betaGP_release_rate_mM_day": release_rate,
        "betaGP_released_current_cycle_mM": beta_gp_cycle,
        "cumulative_phosphate_released_current_cycle_mM": beta_gp_cycle,
        "phosphate_transferred_to_precursor_current_cycle_mM": removed_cycle,
        "mineral_bound_phosphate_equivalent_mM": np.asarray(sim.A, dtype=float) + np.asarray(sim.H, dtype=float),
        "cumulative_precursor_formation_integral": np.asarray(
            [
                np.trapezoid(formation_rate[: i + 1], t[: i + 1])
                for i in range(len(t))
            ],
            dtype=float,
        ),
        "medium_cycle_number": cycle_index + 1,
        "soluble_phosphate_balance_residual_mM": balance_error,
        "max_abs_balance_error": max_abs_error,
    }


def export_phosphate_schaeck_comparison(
    out_dir: Path,
    sim: Any,
    params: dict[str, float],
    media_interval: float,
    figure_name: str = "phosphate_accounting_schaeck_inspired",
    csv_name: str = "phosphate_accounting_schaeck_inspired.csv",
    figure_formats: tuple[str, ...] = ("pdf", "png"),
) -> float:
    out_dir.mkdir(parents=True, exist_ok=True)
    account = phosphate_cycle_accounting(sim, params, media_interval)
    t = np.asarray(sim.t, dtype=float)
    total_free_pi = np.asarray(account["free_phosphate_mM"], dtype=float)
    beta_gp_cycle = np.asarray(account["betaGP_released_current_cycle_mM"], dtype=float)
    cycle_free_p0 = np.asarray(account["cycle_start_free_phosphate_mM"], dtype=float)
    beta_gp_remaining = np.asarray(account["betaGP_remaining_mM"], dtype=float)
    release_rate = np.asarray(account["betaGP_release_rate_mM_day"], dtype=float)
    removed_cycle = np.asarray(account["phosphate_transferred_to_precursor_current_cycle_mM"], dtype=float)
    mineralized_phosphate = np.asarray(account["mineral_bound_phosphate_equivalent_mM"], dtype=float)
    cycle_number = np.asarray(account["medium_cycle_number"], dtype=int)

    with (out_dir / csv_name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "time_days",
                "free_phosphate_mM",
                "cycle_start_free_phosphate_mM",
                "betaGP_released_current_cycle_mM",
                "cumulative_phosphate_released_current_cycle_mM",
                "betaGP_remaining_mM",
                "betaGP_release_rate_mM_day",
                "phosphate_transferred_to_precursor_current_cycle_mM",
                "mineral_bound_phosphate_equivalent_mM",
                "cumulative_precursor_formation_integral",
                "soluble_phosphate_balance_residual_mM",
                "medium_cycle_number",
            ],
        )
        writer.writeheader()
        for i in range(len(t)):
            writer.writerow(
                {
                    "time_days": f"{t[i]:.10g}",
                    "free_phosphate_mM": f"{total_free_pi[i]:.10g}",
                    "cycle_start_free_phosphate_mM": f"{cycle_free_p0[i]:.10g}",
                    "betaGP_released_current_cycle_mM": f"{beta_gp_cycle[i]:.10g}",
                    "cumulative_phosphate_released_current_cycle_mM": f"{beta_gp_cycle[i]:.10g}",
                    "betaGP_remaining_mM": f"{beta_gp_remaining[i]:.10g}",
                    "betaGP_release_rate_mM_day": f"{release_rate[i]:.10g}",
                    "phosphate_transferred_to_precursor_current_cycle_mM": f"{removed_cycle[i]:.10g}",
                    "mineral_bound_phosphate_equivalent_mM": f"{mineralized_phosphate[i]:.10g}",
                    "cumulative_precursor_formation_integral": f"{float(account['cumulative_precursor_formation_integral'][i]):.10g}",
                    "soluble_phosphate_balance_residual_mM": f"{float(account['soluble_phosphate_balance_residual_mM'][i]):.10g}",
                    "medium_cycle_number": int(cycle_number[i]),
                }
            )

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 6.0), sharex=True, constrained_layout=True)
    ax_free, ax_cycle = axes
    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#e6e6e6", linewidth=0.6, alpha=0.8)
        for day in sim.media_days:
            ax.axvline(day, color="#b8c2cc", linestyle="--", linewidth=0.75, alpha=0.55, zorder=0)

    ax_free.plot(t, total_free_pi, color="#1f4e79", linewidth=1.7, label="Total free Pi")
    ax_free.axhline(float(params["P_ext"]), color="#555555", linestyle="--", linewidth=1.0, label="Basal medium Pi")
    ax_free.set_ylabel("Effective reactive Pi (mM)")
    ax_free.set_title("Reactive phosphate accounting", loc="left", pad=6)
    ax_free.legend(frameon=False, loc="best")

    ax_cycle.plot(t, beta_gp_cycle, color="#2a9d8f", linewidth=1.7, label="Cumulative β-GP Pi released this cycle")
    ax_cycle.plot(t, removed_cycle, color="#7a4f9a", linewidth=1.35, linestyle="--", label="Pi transferred to precursor")
    ax_cycle.set_ylabel("Pi released per medium cycle (mM)")
    ax_cycle.set_xlabel("Time (days)")
    ax_cycle.set_title("Cycle-resolved first-order β-GP hydrolysis", loc="left", pad=6)
    ax_cycle.legend(frameon=False, loc="best")

    axes[-1].set_xlim(float(np.min(t)), float(np.max(t)))
    figure_stem = Path(figure_name).stem
    save_figure(fig, out_dir, figure_stem, figure_formats, dpi=300)

    return float(account["max_abs_balance_error"])
