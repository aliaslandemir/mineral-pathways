"""Archived extended publication diagnostics."""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

from .diagnostics import mass_balance_diagnostics
from .io_utils import save_figure
from .model import SimulationResult, simulate


def _continuous_cycle_indices(sim: SimulationResult, cycle: int) -> np.ndarray:
    return np.where(sim.cycle_index == cycle)[0]


def export_baseline_trajectory(path: Path, sim: SimulationResult) -> None:
    fields = ["time_days", "cycle_index", "reset_status", "C", "P", "B", "A", "H", "J_hyd", "S_A", "R_f", "R_m"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for i in range(len(sim.t)):
            writer.writerow({
                "time_days": sim.t[i], "cycle_index": sim.cycle_index[i], "reset_status": sim.reset_status[i],
                "C": sim.C[i], "P": sim.P[i], "B": sim.B[i], "A": sim.A[i], "H": sim.H[i],
                "J_hyd": sim.J_bgp[i], "S_A": sim.S_A[i], "R_f": sim.Rf[i], "R_m": sim.Rm[i],
            })


def export_cycle_flux_integrals(path: Path, sim: SimulationResult) -> None:
    rows = []
    for cycle in np.unique(sim.cycle_index):
        idx = _continuous_cycle_indices(sim, int(cycle))
        if idx.size < 2:
            continue
        integrate = lambda values: float(np.trapezoid(np.asarray(values)[idx], sim.t[idx]))
        rows.append({
            "cycle_index": int(cycle), "start_days": float(sim.t[idx[0]]), "end_days": float(sim.t[idx[-1]]),
            "integral_J_hyd_mM": integrate(sim.J_bgp), "integral_R_f_mM": integrate(sim.Rf),
            "integral_R_m_mM": integrate(sim.Rm), "B_depletion_mM": float(sim.B[idx[0]] - sim.B[idx[-1]]),
        })
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def export_reproducibility_files(out_dir: Path, params: dict[str, Any], seed: int, sim: SimulationResult) -> None:
    clean = {k: v for k, v in params.items() if not str(k).startswith("_")}
    (out_dir / "parameters.json").write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "random_seed.txt").write_text(f"{int(seed)}\n", encoding="utf-8")
    versions = {
        "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
        "scipy": scipy.__version__, "matplotlib": matplotlib.__version__,
    }
    (out_dir / "software_versions.txt").write_text("".join(f"{k}: {v}\n" for k, v in versions.items()), encoding="utf-8")
    export_baseline_trajectory(out_dir / "baseline_trajectories.csv", sim)
    export_cycle_flux_integrals(out_dir / "cycle_flux_integrals.csv", sim)


def export_long_term_boundedness(path: Path, params: dict[str, Any], dt: float, media_interval: float, tolerance: float = 1e-4) -> None:
    rows = []
    for duration in (28.0, 56.0, 84.0, 112.0):
        sim = simulate(params, t_end=duration, dt=dt, media_interval=media_interval)
        complete = []
        for cycle in np.unique(sim.cycle_index):
            cycle_idx = _continuous_cycle_indices(sim, int(cycle))
            if cycle_idx.size > 1 and sim.t[cycle_idx[-1]] - sim.t[cycle_idx[0]] >= media_interval - 1e-8:
                complete.append((int(cycle), cycle_idx))
        selected = complete[-3:]
        total_time = sum(float(sim.t[idx[-1]] - sim.t[idx[0]]) for _, idx in selected)
        int_rf = sum(float(np.trapezoid(sim.Rf[idx], sim.t[idx])) for _, idx in selected)
        int_rm = sum(float(np.trapezoid(sim.Rm[idx], sim.t[idx])) for _, idx in selected)
        mean_rf = int_rf / total_time; mean_rm = int_rm / total_time; net = mean_rf - mean_rm
        rows.append({
            "duration_days": duration, "A_final": sim.A[-1], "H_final": sim.H[-1],
            "A_over_H": float(sim.A[-1] / sim.H[-1]) if sim.H[-1] > 0 else np.nan,
            "mean_net_A_accumulation_final_3_cycles": net, "mean_R_f_final_3_cycles": mean_rf,
            "mean_R_m_final_3_cycles": mean_rm, "quasi_steady_A": abs(net) < tolerance,
            "final_complete_cycles": ";".join(str(cycle) for cycle, _ in selected),
        })
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)


def export_sweep_mechanism_diagnostics(
    out_dir: Path, name: str, sweep: Iterable[tuple[float, SimulationResult, dict[str, Any]]], plot: Any
) -> None:
    sweep = list(sweep)
    variables = (("C", "C (mM)"), ("P", "Effective reactive Pi (mM)"), ("S_A", "Effective formation index"),
                 ("Rf", "$R_f$ (model units day$^{-1}$)"), ("A", "ACP-like precursor (model units)"),
                 ("Rm", "$R_m$ (model units day$^{-1}$)"), ("H", "Apatite-like mineral (model units)"))
    fig, axes = plt.subplots(4, 2, figsize=(10.5, 12.0), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(sweep)))
    rows = []
    for color, (level, sim, _) in zip(colors, sweep):
        for ax, (field, ylabel) in zip(axes.ravel(), variables):
            ax.plot(sim.t, getattr(sim, field), color=color, lw=1.15, label=f"{name}={level:g}")
            ax.set_ylabel(ylabel); ax.set_xlabel("Time (days)"); ax.grid(axis="y", alpha=0.25)
        rows.append({
            "parameter": name, "level": level, "C_min": np.min(sim.C), "P_min": np.min(sim.P),
            "max_S_A": np.max(sim.S_A), "max_R_f": np.max(sim.Rf), "A_final": sim.A[-1],
            "max_R_m": np.max(sim.Rm), "H_final": sim.H[-1],
        })
    axes.ravel()[2].axhline(1.0, color="#a23b3b", ls=":", lw=1.0)
    axes.ravel()[0].legend(frameon=False, ncol=2, fontsize=7, loc="upper center", bbox_to_anchor=(1.05, 1.28))
    axes.ravel()[-1].axis("off")
    save_figure(fig, out_dir, f"diagnostic_{name}_mechanisms", plot.figure_formats, dpi=plot.save_dpi)
    with (out_dir / f"diagnostic_{name}_mechanisms.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)


def generate_publication_figures(out_dir: Path, sim: SimulationResult, params: dict[str, Any], diag: dict[str, Any], plot: Any) -> None:
    # Figure 1: operational model scheme.
    fig, ax = plt.subplots(figsize=(9.2, 3.5), constrained_layout=True); ax.axis("off")
    boxes = [(0.05, "Medium\nC, P, B"), (0.30, "β-GP pool\nB"), (0.52, "Reactive Pi\nP"), (0.72, "ACP-like\nprecursor A"), (0.91, "Apatite-like\nmineral H")]
    for x, label in boxes:
        ax.text(x, 0.55, label, ha="center", va="center", transform=ax.transAxes,
                bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#444444", "lw": 1.0})
    for x0, x1, label in ((0.12, .25, "renewal"), (.36, .47, "$J_{hyd}$"), (.58, .67, "$R_f$"), (.79, .86, "$R_m$")):
        ax.annotate("", (x1, .55), (x0, .55), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.3})
        ax.text((x0+x1)/2, .64, label, ha="center", transform=ax.transAxes)
    ax.text(.78, .20, "Ascorbate\nmatrix support", ha="center", transform=ax.transAxes)
    ax.annotate("", (.83, .46), (.78, .26), xycoords="axes fraction", arrowprops={"arrowstyle": "->", "lw": 1.0, "ls": ":"})
    save_figure(fig, out_dir, "publication_fig1_model_scheme", plot.figure_formats, dpi=plot.save_dpi)

    cumulative_release = np.zeros_like(sim.t)
    for cycle in np.unique(sim.cycle_index):
        idx = _continuous_cycle_indices(sim, int(cycle))
        if idx.size: cumulative_release[idx] = sim.B[idx[0]] - sim.B[idx]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    axes[0,0].plot(sim.t, sim.B, label="β-GP, B"); axes[0,0].plot(sim.t, cumulative_release, label="Cycle Pi released")
    axes[0,1].plot(sim.t, sim.C, label="Reactive Ca"); axes[0,1].plot(sim.t, sim.P, label="Effective reactive Pi")
    axes[1,0].plot(sim.t, sim.S_A, label="$S_A$"); axes[1,0].axhline(1, color="#a23b3b", ls=":", label="Formation threshold")
    axes[1,1].plot(sim.t, sim.A, label="ACP-like precursor"); axes[1,1].plot(sim.t, sim.H, label="Apatite-like mineral")
    for ax in axes.ravel(): ax.set_xlabel("Time (days)"); ax.legend(frameon=False); ax.grid(axis="y", alpha=.25)
    axes[0,0].set_ylabel("Concentration (mM)"); axes[0,1].set_ylabel("Concentration (mM)")
    axes[1,0].set_ylabel("Effective precursor-formation index"); axes[1,1].set_ylabel("Mineral pool (model units)")
    save_figure(fig, out_dir, "publication_fig2_baseline_dynamics", plot.figure_formats, dpi=plot.save_dpi)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), constrained_layout=True)
    axes[0,0].plot(sim.t, sim.J_bgp); axes[0,0].set_ylabel("$J_{hyd}$ (mM day$^{-1}$)")
    axes[0,1].plot(sim.t, sim.Rf); axes[0,1].set_ylabel("$R_f$ (model units day$^{-1}$)")
    axes[1,0].plot(sim.t, sim.Rm); axes[1,0].set_ylabel("$R_m$ (model units day$^{-1}$)")
    axes[1,1].plot(sim.t, diag["phosphate_residual_mM"], label="Phosphate"); axes[1,1].plot(sim.t, diag["calcium_residual_mM"], label="Calcium")
    axes[1,1].set_ylabel("Within-cycle balance residual (mM)"); axes[1,1].legend(frameon=False)
    for ax in axes.ravel(): ax.set_xlabel("Time (days)"); ax.grid(axis="y", alpha=.25)
    save_figure(fig, out_dir, "publication_fig3_fluxes_balances", plot.figure_formats, dpi=plot.save_dpi)
