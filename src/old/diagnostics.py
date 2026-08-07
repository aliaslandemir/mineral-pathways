"""Archived mass-balance diagnostic exports."""

from __future__ import annotations

from pathlib import Path
import csv
from typing import Any

import numpy as np

from .model import validate_simulation_state


def mass_balance_diagnostics(sim: Any, params: dict[str, float]) -> dict[str, Any]:
    """Return cycle-aware calcium and phosphate balance residuals in mM."""
    validate_simulation_state(sim)
    t = np.asarray(sim.t, dtype=float)
    cycle_index = np.asarray(sim.cycle_index, dtype=int)
    cycle_start = np.asarray(sim.cycle_start, dtype=float)
    reset_status = np.asarray(sim.reset_status, dtype=object)

    phosphate_residual = np.zeros_like(t)
    calcium_residual = np.zeros_like(t)
    rows: list[dict[str, float | int | str]] = []

    for cycle in np.unique(cycle_index):
        idx = np.where(cycle_index == cycle)[0]
        idx = idx[reset_status[idx] != "pre_reset"]
        if idx.size == 0:
            continue
        i0 = int(idx[0])
        p0 = float(sim.P[i0] + sim.B[i0] + sim.A[i0] + sim.H[i0])
        ca0 = float(sim.C[i0] + params["r_A"] * sim.A[i0] + params["r_H"] * sim.H[i0])
        local_p = []
        local_ca = []
        for i in idx:
            p_expected = p0
            ca_expected = ca0
            phosphate_residual[i] = float(sim.P[i] + sim.B[i] + sim.A[i] + sim.H[i] - p_expected)
            calcium_residual[i] = float(sim.C[i] + params["r_A"] * sim.A[i] + params["r_H"] * sim.H[i] - ca_expected)
            local_p.append(phosphate_residual[i])
            local_ca.append(calcium_residual[i])
        rows.append(
            {
                "cycle_index": int(cycle),
                "cycle_start_days": float(cycle_start[i0]),
                "max_abs_phosphate_residual_mM": float(np.max(np.abs(local_p))),
                "max_rel_phosphate_residual": float(np.max(np.abs(local_p)) / max(abs(p0), 1.0)),
                "max_abs_calcium_residual_mM": float(np.max(np.abs(local_ca))),
                "max_rel_calcium_residual": float(np.max(np.abs(local_ca)) / max(abs(ca0), 1.0)),
            }
        )

    active = reset_status != "pre_reset"
    p_scale = max(1.0, float(np.nanmax(np.abs(sim.P + sim.B + sim.A + sim.H))))
    ca_scale = max(1.0, float(np.nanmax(np.abs(sim.C + params["r_A"] * sim.A + params["r_H"] * sim.H))))
    return {
        "time_days": t,
        "cycle_index": cycle_index,
        "reset_status": reset_status,
        "phosphate_residual_mM": phosphate_residual,
        "calcium_residual_mM": calcium_residual,
        "cycle_rows": rows,
        "max_abs_phosphate_residual_mM": float(np.max(np.abs(phosphate_residual[active]))),
        "max_rel_phosphate_residual": float(np.max(np.abs(phosphate_residual[active])) / p_scale),
        "max_abs_calcium_residual_mM": float(np.max(np.abs(calcium_residual[active]))),
        "max_rel_calcium_residual": float(np.max(np.abs(calcium_residual[active])) / ca_scale),
    }


def export_mass_balance_diagnostics(out_dir: Path, sim: Any, params: dict[str, float], csv_name: str = "mass_balance_diagnostics.csv") -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    diag = mass_balance_diagnostics(sim, params)
    with (out_dir / csv_name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "time_days",
                "cycle_index",
                "reset_status",
                "phosphate_total_mM",
                "calcium_total_mM",
                "phosphate_residual_mM",
                "relative_phosphate_residual",
                "calcium_residual_mM",
                "relative_calcium_residual",
            ],
        )
        writer.writeheader()
        for i in range(len(diag["time_days"])):
            writer.writerow(
                {
                    "time_days": f"{float(diag['time_days'][i]):.10g}",
                    "cycle_index": int(diag["cycle_index"][i]),
                    "reset_status": str(diag["reset_status"][i]),
                    "phosphate_total_mM": f"{float(sim.P[i] + sim.B[i] + sim.A[i] + sim.H[i]):.10g}",
                    "calcium_total_mM": f"{float(sim.C[i] + params['r_A'] * sim.A[i] + params['r_H'] * sim.H[i]):.10g}",
                    "phosphate_residual_mM": f"{float(diag['phosphate_residual_mM'][i]):.10g}",
                    "relative_phosphate_residual": f"{float(abs(diag['phosphate_residual_mM'][i]) / max(1.0, abs(sim.P[i] + sim.B[i] + sim.A[i] + sim.H[i]))):.10g}",
                    "calcium_residual_mM": f"{float(diag['calcium_residual_mM'][i]):.10g}",
                    "relative_calcium_residual": f"{float(abs(diag['calcium_residual_mM'][i]) / max(1.0, abs(sim.C[i] + params['r_A'] * sim.A[i] + params['r_H'] * sim.H[i]))):.10g}",
                }
            )
    with (out_dir / "mass_balance_cycle_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(diag["cycle_rows"][0].keys()))
        writer.writeheader()
        writer.writerows(diag["cycle_rows"])
    return diag
