"""Competing matrix/bulk calcium-phosphate model.

Run from the repository root::

    python -m src.model \
    --t-end 30 \
    --dt 0.05 \
    --media-interval 2 \
    --K_Q 66.0 \
    --k_b 0.2 \
    --Q_max 1.0 \
    --n_b 1.0 \
    --output output/model/baseline_trajectory.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp


EPS = 1.0e-12
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEW_MODEL_DIR = PROJECT_ROOT
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"
FIXED_MODEL_ASSUMPTIONS = {
    "hydrolysis_law": "J_hyd = k_hyd * B",
    "eta_48h_reference_interval_day": 2.0,
    "renewal_fraction": 1.0,
    "matrix_index": "S_A = C^x * P^y / K_A",
    "solution_index": "S_Q = C^x * P^y / K_Q",
    "matrix_law": "R_f = k_f * [S_A - 1]_+ * [M_max - (A + H)]_+",
    "maturation_law": "R_m = k_m * A",
    "solution_law": "R_b = k_b * [S_Q - 1]_+^n_b * [Q_max - Q]_+",
    "cli_alias": "--K_H is accepted as a command-line alias for K_Q",
}


def resolve_output_path(path: Path) -> Path:
    """Resolve relative output paths from the repository root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class ModelParameters:
    """Physical and phenomenological parameters in the revised manuscript."""

    C_ext: float = 2.00
    P_ext: float = 1.25
    B_ext: float = 10.0
    eta_48h: float = 0.30
    x: float = 3.0
    y: float = 2.0
    K_A: float = 33.0
    K_Q: float = 66.0
    k_f: float = 0.2
    M_max: float = 1.0
    k_m: float = 0.2
    k_b: float = 0.2
    Q_max: float = 1.0
    n_b: float = 1.0
    r_A: float = 1.245
    r_H: float = 1.585

    @property
    def k_hyd(self) -> float:
        """First-order hydrolysis rate corresponding to ``eta_48h``."""
        return -np.log1p(-self.eta_48h) / 2.0

    @property
    def K_H(self) -> float:
        """Backward-compatible alias; the manuscript parameter is K_Q."""
        return self.K_Q

    def changed(self, **changes: float) -> "ModelParameters":
        if "K_H" in changes:
            changes = dict(changes)
            changes["K_Q"] = changes.pop("K_H")
        return replace(self, **changes)

    def validate(self) -> None:
        positive = {
            "x": self.x,
            "y": self.y,
            "K_A": self.K_A,
            "K_Q": self.K_Q,
            "M_max": self.M_max,
            "Q_max": self.Q_max,
            "n_b": self.n_b,
            "r_A": self.r_A,
            "r_H": self.r_H,
        }
        for name, value in positive.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.B_ext < 0.0 or self.C_ext < 0.0 or self.P_ext < 0.0:
            raise ValueError("External soluble concentrations must be non-negative")
        if self.k_f < 0.0 or self.k_m < 0.0 or self.k_b < 0.0:
            raise ValueError("k_f, k_m, and k_b must be non-negative")
        if self.K_Q <= self.K_A:
            raise ValueError("K_Q must be greater than K_A so solution-driven onset occurs after matrix-associated onset")
        if not 0.0 <= self.eta_48h < 1.0:
            raise ValueError("eta_48h must be in [0, 1)")
        if self.r_H < self.r_A:
            raise ValueError("r_H must be at least r_A")


@dataclass(frozen=True)
class SimulationSettings:
    t_end: float = 30.0
    dt: float = 0.05
    media_interval: float = 2.0
    method: str = "RK45"
    rtol: float = 1.0e-6
    atol: float = 1.0e-9

    def validate(self) -> None:
        if self.t_end <= 0.0 or self.dt <= 0.0 or self.media_interval <= 0.0:
            raise ValueError("t_end, dt, and media_interval must be positive")


@dataclass(frozen=True)
class StochasticSettings:
    n_runs: int = 100
    seed: int = 42
    r_A_min: float = 1.00
    r_A_max: float = 1.49
    r_H_min: float = 1.50
    r_H_max: float = 1.67

    def validate(self) -> None:
        if self.n_runs < 1:
            raise ValueError("n_runs must be at least 1")
        if self.r_A_min <= 0.0 or self.r_A_max < self.r_A_min:
            raise ValueError("invalid r_A range")
        if self.r_H_min <= self.r_A_max or self.r_H_max < self.r_H_min:
            raise ValueError("r_H range must be ordered and entirely above r_A")


@dataclass
class SimulationResult:
    t: np.ndarray
    B: np.ndarray
    C: np.ndarray
    P: np.ndarray
    A: np.ndarray
    H: np.ndarray
    Q: np.ndarray
    J_hyd: np.ndarray
    R_f: np.ndarray
    R_m: np.ndarray
    R_b: np.ndarray
    S_A: np.ndarray
    S_Q: np.ndarray
    cycle: np.ndarray
    reset_status: np.ndarray
    parameters: ModelParameters
    settings: SimulationSettings

    @property
    def M_mat(self) -> np.ndarray:
        return self.A + self.H

    @property
    def M_tot(self) -> np.ndarray:
        return self.A + self.H + self.Q

    @property
    def S_H(self) -> np.ndarray:
        """Backward-compatible alias; the manuscript symbol is S_Q."""
        return self.S_Q

    @property
    def phosphate_inventory(self) -> np.ndarray:
        return self.B + self.P + self.A + self.H + self.Q

    @property
    def calcium_inventory(self) -> np.ndarray:
        p = self.parameters
        return self.C + p.r_A * self.A + p.r_H * (self.H + self.Q)

    @property
    def matrix_formation_fraction(self) -> np.ndarray:
        total_rate = self.R_f + self.R_b
        return np.divide(self.R_f, total_rate, out=np.full_like(total_rate, np.nan), where=total_rate > EPS)

    @property
    def final_matrix_fraction(self) -> float:
        return float(self.M_mat[-1] / self.M_tot[-1]) if self.M_tot[-1] > EPS else np.nan

    def values_at(self, days: np.ndarray, field: str) -> np.ndarray:
        values = np.asarray(getattr(self, field), dtype=float)
        keep = self.reset_status != "pre_reset"
        return np.interp(np.asarray(days, dtype=float), self.t[keep], values[keep])

    def to_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "t", "B", "C", "P", "A", "H", "Q", "M_mat", "M_tot",
            "S_A", "S_Q", "J_hyd", "R_f", "R_m", "R_b", "cycle", "reset_status",
        ]
        arrays = {name: getattr(self, name) for name in fields}
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(zip(*(arrays[name] for name in fields)))


class MineralizationModel:
    """Six-state ODE model with discrete medium-renewal events."""

    state_names = ("B", "C", "P", "A", "H", "Q")

    def __init__(self, parameters: ModelParameters | None = None):
        self.p = parameters or ModelParameters()
        self.p.validate()

    def formation_index(self, C: np.ndarray | float, P: np.ndarray | float) -> np.ndarray | float:
        return np.asarray(C) ** self.p.x * np.asarray(P) ** self.p.y / self.p.K_A

    def apatite_index(self, C: np.ndarray | float, P: np.ndarray | float) -> np.ndarray | float:
        return self.solution_index(C, P)

    def solution_index(self, C: np.ndarray | float, P: np.ndarray | float) -> np.ndarray | float:
        return np.asarray(C) ** self.p.x * np.asarray(P) ** self.p.y / self.p.K_Q

    def hydrolysis_rate(self, B: np.ndarray | float) -> np.ndarray | float:
        B_value = np.maximum(np.asarray(B, dtype=float), 0.0)
        return self.p.k_hyd * B_value

    def matrix_capacity_factor(self, M_mat: np.ndarray | float) -> np.ndarray | float:
        """Remaining matrix-associated mineralization capacity."""
        return np.maximum(1.0 - np.asarray(M_mat, dtype=float) / self.p.M_max, 0.0)

    def matrix_free_capacity(self, M_mat: np.ndarray | float) -> np.ndarray | float:
        return np.maximum(self.p.M_max - np.asarray(M_mat, dtype=float), 0.0)

    def bulk_free_capacity(self, Q: np.ndarray | float) -> np.ndarray | float:
        return np.maximum(self.p.Q_max - np.asarray(Q, dtype=float), 0.0)

    def rates(self, state: np.ndarray, availability_dt: float | None = None) -> tuple[float, float, float, float, float, float]:
        B, C, P, A, H, Q = np.maximum(np.asarray(state, dtype=float), 0.0)
        S_A = float(self.formation_index(C, P))
        S_Q = float(self.solution_index(C, P))
        M_mat = A + H
        matrix_capacity = float(self.matrix_free_capacity(M_mat))
        bulk_capacity = float(self.bulk_free_capacity(Q))
        R_f = self.p.k_f * max(S_A - 1.0, 0.0) * matrix_capacity
        R_b = self.p.k_b * max(S_Q - 1.0, 0.0) ** self.p.n_b * bulk_capacity
        R_m = self.p.k_m * A
        J_hyd = float(self.hydrolysis_rate(B))

        # At a depleted boundary, disable reactions that require that pool.
        if B <= 0.0:
            J_hyd = 0.0
        if P <= 0.0:
            R_f = 0.0
            R_b = 0.0
        if C <= 0.0:
            R_f = 0.0
            R_m = 0.0
            R_b = 0.0
        if A <= 0.0:
            R_m = 0.0
        if availability_dt is not None and availability_dt > 0.0:
            p_consumption = R_f + R_b
            c_consumption = self.p.r_A * R_f + (self.p.r_H - self.p.r_A) * R_m + self.p.r_H * R_b
            scale = 1.0
            if p_consumption > 0.0:
                scale = min(scale, P / (p_consumption * availability_dt))
            if c_consumption > 0.0:
                scale = min(scale, C / (c_consumption * availability_dt))
            if scale < 1.0:
                R_f *= scale
                R_m *= scale
                R_b *= scale
        return J_hyd, R_f, R_m, R_b, S_A, S_Q

    def rhs(self, t: float, state: np.ndarray, cycle_start: float, availability_dt: float) -> np.ndarray:
        J_hyd, R_f, R_m, R_b, _, _ = self.rates(state, availability_dt)
        p = self.p
        return np.array(
            [
                -J_hyd,
                -p.r_A * R_f - (p.r_H - p.r_A) * R_m - p.r_H * R_b,
                J_hyd - R_f - R_b,
                R_f - R_m,
                R_m,
                R_b,
            ]
        )

    def simulate(self, settings: SimulationSettings | None = None) -> SimulationResult:
        settings = settings or SimulationSettings()
        settings.validate()
        p = self.p
        renewal_days = np.arange(settings.media_interval, settings.t_end, settings.media_interval)
        endpoints = np.append(renewal_days, settings.t_end)
        state = np.array([p.B_ext, p.C_ext, p.P_ext, 0.0, 0.0, 0.0], dtype=float)
        segment_start = 0.0
        times: list[np.ndarray] = []
        states: list[np.ndarray] = []
        cycles: list[np.ndarray] = []
        statuses: list[np.ndarray] = []

        for cycle_index, segment_end in enumerate(endpoints):
            n_points = max(2, int(np.ceil((segment_end - segment_start) / settings.dt)) + 1)
            t_eval = np.linspace(segment_start, segment_end, n_points)
            solution = solve_ivp(
                self.rhs,
                (segment_start, segment_end),
                state,
                args=(segment_start, settings.dt),
                t_eval=t_eval,
                method=settings.method,
                rtol=settings.rtol,
                atol=settings.atol,
                max_step=settings.dt,
            )
            if not solution.success:
                raise RuntimeError(solution.message)
            segment_t = solution.t if not times else solution.t[1:]
            segment_y = solution.y if not states else solution.y[:, 1:]
            if np.min(segment_y) < -1.0e-4:
                raise RuntimeError(f"ODE solution left the physical state domain: minimum={np.min(segment_y):.3g}")
            segment_y = np.maximum(segment_y, 0.0)
            status = np.full(segment_t.size, "continuous", dtype=object)
            is_renewal = np.any(np.isclose(segment_end, renewal_days))
            if is_renewal:
                status[-1] = "pre_reset"
            times.append(segment_t)
            states.append(segment_y)
            cycles.append(np.full(segment_t.size, cycle_index, dtype=int))
            statuses.append(status)
            state = np.maximum(solution.y[:, -1], 0.0)

            if is_renewal:
                external = np.array([p.B_ext, p.C_ext, p.P_ext])
                state[:3] = external
                times.append(np.array([segment_end]))
                states.append(state[:, None])
                cycles.append(np.array([cycle_index + 1]))
                statuses.append(np.array(["post_reset"], dtype=object))
            segment_start = float(segment_end)

        t = np.concatenate(times)
        trajectory = np.concatenate(states, axis=1)
        cycle = np.concatenate(cycles)
        reset_status = np.concatenate(statuses)
        B, C, P, A, H, Q = trajectory
        S_A = np.asarray(self.formation_index(C, P))
        S_Q = np.asarray(self.solution_index(C, P))
        J_hyd = np.asarray(self.hydrolysis_rate(B))
        matrix_capacity = np.asarray(self.matrix_free_capacity(A + H))
        bulk_capacity = np.asarray(self.bulk_free_capacity(Q))
        R_f = p.k_f * np.maximum(S_A - 1.0, 0.0) * matrix_capacity
        R_b = p.k_b * np.maximum(S_Q - 1.0, 0.0) ** p.n_b * bulk_capacity
        R_m = np.where(C > 0.0, p.k_m * A, 0.0)
        p_consumption = R_f + R_b
        c_consumption = p.r_A * R_f + (p.r_H - p.r_A) * R_m + p.r_H * R_b
        scale = np.ones_like(R_f)
        scale = np.minimum(scale, np.divide(P, p_consumption * settings.dt, out=np.ones_like(P), where=p_consumption > 0.0))
        scale = np.minimum(scale, np.divide(C, c_consumption * settings.dt, out=np.ones_like(C), where=c_consumption > 0.0))
        R_f *= scale
        R_m *= scale
        R_b *= scale
        return SimulationResult(
            t, B, C, P, A, H, Q, J_hyd, R_f, R_m, R_b, S_A, S_Q,
            cycle, reset_status, p, settings,
        )

    def threshold_times(self) -> tuple[float, float]:
        """Analytical early-cycle crossing times ``t_A`` and ``t_Q``."""
        p = self.p
        if p.C_ext <= 0.0:
            return np.inf, np.inf
        P_A = (p.K_A / p.C_ext**p.x) ** (1.0 / p.y)
        P_Q = (p.K_Q / p.C_ext**p.x) ** (1.0 / p.y)

        def crossing(target: float) -> float:
            release = target - p.P_ext
            if release <= 0.0:
                return 0.0
            if release >= p.B_ext or p.k_hyd <= 0.0:
                return np.inf
            remaining = p.B_ext - release
            return float(-np.log(remaining / p.B_ext) / p.k_hyd)

        return crossing(P_A), crossing(P_Q)


def sample_stoichiometries(base: ModelParameters, stochastic: StochasticSettings) -> list[ModelParameters]:
    """Sample fixed-within-run Ca/P ratios for an ensemble."""
    stochastic.validate()
    rng = np.random.default_rng(stochastic.seed)
    r_a = rng.uniform(stochastic.r_A_min, stochastic.r_A_max, stochastic.n_runs)
    r_h = rng.uniform(stochastic.r_H_min, stochastic.r_H_max, stochastic.n_runs)
    return [base.changed(r_A=float(a), r_H=float(h)) for a, h in zip(r_a, r_h)]


def simulate_ensemble(base: ModelParameters, settings: SimulationSettings, stochastic: StochasticSettings) -> list[SimulationResult]:
    return [MineralizationModel(parameters).simulate(settings) for parameters in sample_stoichiometries(base, stochastic)]


def save_ensemble(runs: list[SimulationResult], output_dir: Path) -> None:
    """Save sampled ratios and pointwise ensemble summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "stoichiometry_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "r_A", "r_H"])
        writer.writerows((i, run.parameters.r_A, run.parameters.r_H) for i, run in enumerate(runs))
    fields = ("B", "C", "P", "A", "H", "Q", "M_mat", "M_tot", "S_A", "S_Q", "R_f", "R_m", "R_b")
    with (output_dir / "trajectory_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["time_day"]
        for field in fields:
            header.extend((f"{field}_mean", f"{field}_sd", f"{field}_q025", f"{field}_median", f"{field}_q975"))
        writer.writerow(header)
        arrays = {field: np.asarray([getattr(run, field) for run in runs]) for field in fields}
        for index, time in enumerate(runs[0].t):
            row: list[float] = [float(time)]
            for field in fields:
                values = arrays[field][:, index]
                row.extend((float(np.mean(values)), float(np.std(values)), *np.quantile(values, [0.025, 0.5, 0.975])))
            writer.writerow(row)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every model and numerical setting to an argument parser."""
    parser.add_argument("--t-end", type=float, default=30.0, help="Simulation duration in days.")
    parser.add_argument("--dt", type=float, default=0.05, help="Maximum solver step and recorded spacing in days.")
    parser.add_argument("--media-interval", type=float, default=2.0, help="Time between medium renewals in days.")
    parser.add_argument("--solver-method", dest="method", default="RK45", help="SciPy solve_ivp integration method.")
    parser.add_argument("--solver-rtol", dest="rtol", type=float, default=1.0e-6, help="Relative ODE solver tolerance.")
    parser.add_argument("--solver-atol", dest="atol", type=float, default=1.0e-9, help="Absolute ODE solver tolerance.")
    parser.add_argument("--C_ext", dest="C_ext", type=float, default=2.00, help="Fresh-medium soluble calcium concentration (mM).")
    parser.add_argument("--P_ext", dest="P_ext", type=float, default=1.25, help="Fresh-medium inorganic phosphate concentration (mM).")
    parser.add_argument("--beta_GP", dest="B_ext", type=float, default=10.0, help="Fresh-medium beta-glycerophosphate concentration (mM).")
    parser.add_argument("--eta_48h", dest="eta_48h", type=float, default=0.30, help="Fraction of beta-GP hydrolyzed over 48 hours.")
    parser.add_argument("--calcium-exponent", dest="x", type=float, default=3.0, help="Calcium exponent x in S_A=C^x P^y/K_A.")
    parser.add_argument("--phosphate-exponent", dest="y", type=float, default=2.0, help="Phosphate exponent y in S_A=C^x P^y/K_A.")
    parser.add_argument("--K_A", dest="K_A", type=float, default=33.0, help="Operational matrix-pathway formation parameter.")
    parser.add_argument("--K_Q", "--K_H", dest="K_Q", type=float, default=66.0, help="Operational solution-driven onset parameter K_Q; --K_H is accepted as an alias.")
    parser.add_argument("--k_f", dest="k_f", type=float, default=0.2, help="Matrix precursor-formation rate constant (1/day).")
    parser.add_argument("--M_max", dest="M_max", type=float, default=1.0, help="Matrix-associated mineral capacity (mM P-equivalent).")
    parser.add_argument("--k_m", dest="k_m", type=float, default=0.2, help="ACP-like precursor maturation rate constant (1/day).")
    parser.add_argument("--k_b", dest="k_b", type=float, default=0.2, help="Solution-driven mineral-retention coefficient (1/day).")
    parser.add_argument("--Q_max", dest="Q_max", type=float, default=1.0, help="Retained solution-origin mineral capacity (mM P-equivalent).")
    parser.add_argument("--n_b", dest="n_b", type=float, default=1.0, help="Solution-driven activation steepness exponent.")


def add_stochastic_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-runs", "--n_runs", type=int, default=100, help="Number of stochastic Ca/P-ratio realizations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible Ca/P-ratio sampling.")
    parser.add_argument("--r_A_min", dest="r_A_min", type=float, default=1.00, help="Minimum sampled ACP-like Ca/P ratio.")
    parser.add_argument("--r_A_max", dest="r_A_max", type=float, default=1.49, help="Maximum sampled ACP-like Ca/P ratio.")
    parser.add_argument("--r_H_min", dest="r_H_min", type=float, default=1.50, help="Minimum sampled apatite-like Ca/P ratio.")
    parser.add_argument("--r_H_max", dest="r_H_max", type=float, default=1.67, help="Maximum sampled apatite-like Ca/P ratio.")


def parameters_from_args(args: argparse.Namespace, stochastic: StochasticSettings | None = None) -> ModelParameters:
    names = set(asdict(ModelParameters()))
    values = {name: value for name, value in vars(args).items() if name in names and value is not None}
    if "K_H" in vars(args) and "K_Q" not in values:
        values["K_Q"] = vars(args)["K_H"]
    if stochastic is not None:
        values["r_A"] = 0.5 * (stochastic.r_A_min + stochastic.r_A_max)
        values["r_H"] = 0.5 * (stochastic.r_H_min + stochastic.r_H_max)
    return ModelParameters(**values)


def settings_from_args(args: argparse.Namespace) -> SimulationSettings:
    return SimulationSettings(args.t_end, args.dt, args.media_interval, args.method, args.rtol, args.atol)


def stochastic_from_args(args: argparse.Namespace) -> StochasticSettings:
    return StochasticSettings(args.n_runs, args.seed, args.r_A_min, args.r_A_max, args.r_H_min, args.r_H_max)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the six-state competing-pathway model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_model_arguments(parser)
    add_stochastic_arguments(parser)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "model" / "baseline_trajectory.csv")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = settings_from_args(args)
    stochastic = stochastic_from_args(args)
    parameters = parameters_from_args(args, stochastic)
    result = MineralizationModel(parameters).simulate(settings)
    output = resolve_output_path(args.output)
    result.to_csv(output)
    runs = simulate_ensemble(parameters, settings, stochastic)
    save_ensemble(runs, output.parent)
    print(f"Saved {output}")
    print(f"Final M_mat={result.M_mat[-1]:.6g}, Q={result.Q[-1]:.6g}, M_tot={result.M_tot[-1]:.6g}")


if __name__ == "__main__":
    main()
