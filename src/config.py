from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .model import (
    ETA_48H,
    K_HYD,
)


@dataclass(frozen=True)
class ModelDefaults:
    C_ext: float = 1.8018
    P_ext: float = 0.906
    beta_GP: float = 10.0
    eta_48h: float = ETA_48H
    k_hyd: float = K_HYD
    alpha: float = 3.0
    beta: float = 2.0
    n: float = 1.0
    K_A: float = 18.0
    k_f: float = 1.0
    M_max: float = 2.0
    k_m: float = 0.2
    a_AA0: float = 0.28
    t_AA_half_life: float = 24.0
    media_renewal_fraction: float = 1.00


@dataclass(frozen=True)
class RunDefaults:
    t_end: float = 30.0
    dt: float = 0.05
    media_interval: float = 2.0
    n_runs: int = 100
    jobs: int = 6
    seed: int = 42
    solver_method: str = "RK45"
    solver_rtol: float = 1e-6
    solver_atol: float = 1e-9


@dataclass(frozen=True)
class StochasticDefaults:
    r_A_min: float = 1.20
    r_A_max: float = 1.49
    r_H_min: float = 1.50
    r_H_max: float = 1.67
    Xi: float = 1.0
    t_AA_half_life_levels: tuple[float, ...] = (6.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0)
    beta_gp_levels: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0)
    eta_48h_levels: tuple[float, ...] = (0.05, 0.10, 0.18, 0.22, 0.25, 0.33, 0.40, 0.50, 0.65)
    K_A_levels: tuple[float, ...] = (8.0, 10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 24.0)
    k_f_levels: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00)
    M_max_levels: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
    k_m_levels: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20, 0.50, 1.00, 2.00)


@dataclass(frozen=True)
class PlotDefaults:
    save_dpi: int = 300
    figure_dpi: int = 180
    font_size: float = 10.0
    axes_title_size: float = 11.0
    axes_label_size: float = 10.5
    legend_font_size: float = 8.8
    tick_label_size: float = 9.5
    phase_lines_days: tuple[float, ...] = ()
    figure_formats: tuple[str, ...] = ("pdf", "png")


@dataclass(frozen=True)
class SensitivityDefaults:
    sobol_bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "media_interval": (1.0, 5.0),
            "beta_GP": (0.0, 20.0),
            "eta_48h": (0.05, 0.65),
            "K_A": (8.0, 24.0),
            "k_f": (0.10, 2.00),
            "M_max": (2.0, 8.0),
            "k_m": (0.01, 2.00),
            "t_AA_half_life": (6.0, 72.0),
        }
    )
    uncertainty_params: tuple[str, ...] = (
        "media_interval",
        "beta_GP",
        "eta_48h",
        "K_A",
        "k_f",
        "M_max",
        "k_m",
        "t_AA_half_life",
    )


MODEL_DEFAULTS = ModelDefaults()
RUN_DEFAULTS = RunDefaults()
STOCHASTIC_DEFAULTS = StochasticDefaults()
PLOT_DEFAULTS = PlotDefaults()
SENSITIVITY_DEFAULTS = SensitivityDefaults()


def build_base_params(defaults: ModelDefaults = MODEL_DEFAULTS) -> dict[str, float | str]:
    params = asdict(defaults)
    params["r_A"] = 0.5 * (STOCHASTIC_DEFAULTS.r_A_min + STOCHASTIC_DEFAULTS.r_A_max)
    params["r_H"] = 0.5 * (STOCHASTIC_DEFAULTS.r_H_min + STOCHASTIC_DEFAULTS.r_H_max)
    params["xi_A"] = 0.5
    params["xi_H"] = 0.5
    params["solver_method"] = RUN_DEFAULTS.solver_method
    params["solver_rtol"] = RUN_DEFAULTS.solver_rtol
    params["solver_atol"] = RUN_DEFAULTS.solver_atol
    return params
