from __future__ import annotations

from typing import Iterable

import numpy as np


def _validate_bounds(r_A_min: float, r_A_max: float, r_H_min: float, r_H_max: float, xi_scale: float) -> None:
    values = [r_A_min, r_A_max, r_H_min, r_H_max, xi_scale]
    if not all(np.isfinite(v) for v in values):
        raise ValueError("Ca/P sampling bounds and variability scale must be finite.")
    if r_A_min > r_A_max:
        raise ValueError("r_A_min must be <= r_A_max.")
    if r_H_min > r_H_max:
        raise ValueError("r_H_min must be <= r_H_max.")
    if not 0.0 <= xi_scale <= 1.0:
        raise ValueError("Ca/P variability scale must be in [0, 1].")
    if r_A_max > r_H_min:
        raise ValueError("Configured Ca/P ranges overlap; expected r_A_max <= r_H_min.")


def scaled_unit_sample(u: float, variability_scale: float) -> float:
    u = min(max(float(u), 0.0), 1.0)
    variability_scale = min(max(float(variability_scale), 0.0), 1.0)
    return 0.5 + variability_scale * (u - 0.5)


def build_ca_p_samples(
    n_runs: int,
    seed: int,
    r_A_min: float,
    r_A_max: float,
    r_H_min: float,
    r_H_max: float,
    xi_scale: float,
) -> list[dict[str, float]]:
    """Sample effective Ca/P ratios for ACP-like and apatite-like pools."""
    if int(n_runs) < 1:
        raise ValueError("n_runs must be at least 1.")
    _validate_bounds(r_A_min, r_A_max, r_H_min, r_H_max, xi_scale)

    rng = np.random.default_rng(int(seed))
    u_a_vals = rng.uniform(0.0, 1.0, size=int(n_runs))
    u_h_vals = rng.uniform(0.0, 1.0, size=int(n_runs))
    xi_a_vals = np.array([scaled_unit_sample(u, xi_scale) for u in u_a_vals], dtype=float)
    xi_h_vals = np.array([scaled_unit_sample(u, xi_scale) for u in u_h_vals], dtype=float)
    r_a_vals = float(r_A_min) + (float(r_A_max) - float(r_A_min)) * xi_a_vals
    r_h_vals = float(r_H_min) + (float(r_H_max) - float(r_H_min)) * xi_h_vals

    return [
        {
            "r_A": float(r_a),
            "r_H": float(r_h),
            "xi_A": float(xi_a),
            "xi_H": float(xi_h),
        }
        for r_a, r_h, xi_a, xi_h in zip(r_a_vals, r_h_vals, xi_a_vals, xi_h_vals)
    ]


def apply_stochastic_variability(params: dict[str, float], sample: dict[str, float]) -> dict[str, float]:
    p = dict(params)
    p.update(sample)
    return p
