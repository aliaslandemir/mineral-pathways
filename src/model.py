from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.integrate import solve_ivp

MIN_CONCENTRATION = 0.0
MIN_K_A = 1e-9
MIN_DENOMINATOR = 1e-12
NEGATIVE_STATE_TOLERANCE = 1e-6
POSTPROCESS_NEGATIVE_STATE_TOLERANCE = 1e-3
ETA_48H = 0.22
ETA_REFERENCE_DAYS = 2.0
K_HYD = -np.log1p(-ETA_48H) / ETA_REFERENCE_DAYS
ASCORBATE_HALF_SUPPORT = 0.14


@dataclass
class SimulationResult:
    t: np.ndarray
    C: np.ndarray
    P: np.ndarray
    B: np.ndarray
    A: np.ndarray
    H: np.ndarray
    J_bgp: np.ndarray
    Rf: np.ndarray
    Rm: np.ndarray
    S_A: np.ndarray
    f_MV: np.ndarray
    f_mat: np.ndarray
    media_days: List[float]
    reset_status: np.ndarray
    cycle_index: np.ndarray
    cycle_start: np.ndarray
    time_since_reset: np.ndarray

    @property
    def dA_dt(self) -> np.ndarray:
        return self.Rf - self.Rm

    @property
    def pre_reset_mask(self) -> np.ndarray:
        return self.reset_status == "pre_reset"

    @property
    def post_reset_mask(self) -> np.ndarray:
        return self.reset_status == "post_reset"

    @property
    def continuous_mask(self) -> np.ndarray:
        return self.reset_status == "continuous"


def phosphate_release_rate(beta_gp: float, k_hyd: float = K_HYD) -> float:
    """Reactive phosphate released from beta-GP, in mM/day."""
    return max(float(k_hyd), 0.0) * max(float(beta_gp), 0.0)


def hydrolysis_fraction_from_rate(k_hyd: float, interval_days: float) -> float:
    """Fraction hydrolyzed over an interval for a first-order rate constant."""
    interval = max(float(interval_days), 0.0)
    return float(-np.expm1(-max(float(k_hyd), 0.0) * interval))


def hydrolysis_rate_from_fraction(fraction: float, interval_days: float) -> float:
    """First-order rate constant equivalent to a fractional interval release."""
    interval = float(interval_days)
    if interval <= 0.0:
        raise ValueError("Hydrolysis fraction interval must be positive.")
    fraction = min(max(float(fraction), 0.0), 1.0 - MIN_DENOMINATOR)
    return float(-np.log1p(-fraction) / interval)


def hydrolysis_fraction_for_dt(
    eta_48h: float,
    dt: float,
    reference_days: float = ETA_REFERENCE_DAYS,
) -> float:
    """Convert a reference hydrolysis fraction to the exact fraction over dt."""
    return hydrolysis_fraction_from_rate(
        hydrolysis_rate_from_fraction(eta_48h, reference_days),
        dt,
    )


def beta_gp_hydrolysis_rate_constant(params: Dict[str, float]) -> float:
    if "k_hyd" in params and bool(params.get("_k_hyd_explicit", False)):
        return max(float(params["k_hyd"]), 0.0)
    if params.get("eta_dt") is not None and bool(params.get("_eta_dt_explicit", False)):
        return hydrolysis_rate_from_fraction(
            float(params["eta_dt"]),
            float(params["eta_dt_interval_days"]),
        )
    eta_48h = float(params.get("eta_48h", ETA_48H))
    return hydrolysis_rate_from_fraction(eta_48h, ETA_REFERENCE_DAYS)


def synchronize_hydrolysis_parameters(params: Dict[str, float], dt: float, media_interval: float | None = None) -> float:
    """Resolve equivalent first-order hydrolysis controls without changing B_ext."""
    k_hyd = beta_gp_hydrolysis_rate_constant(params)
    params["k_hyd"] = k_hyd
    params["eta_48h"] = hydrolysis_fraction_from_rate(k_hyd, ETA_REFERENCE_DAYS)
    params["eta_dt"] = hydrolysis_fraction_from_rate(k_hyd, dt)
    params["eta_dt_interval_days"] = float(dt)
    interval = float(params.get("media_interval", ETA_REFERENCE_DAYS) if media_interval is None else media_interval)
    params["beta_GP_cycle"] = max(float(params["beta_GP"]), 0.0)
    params["beta_GP_cycle_release"] = params["beta_GP_cycle"] * hydrolysis_fraction_from_rate(k_hyd, interval)
    return k_hyd


def beta_gp_available_phosphate(params: Dict[str, float], duration_days: float | None = None) -> float:
    duration = float(params.get("media_interval", 0.0) if duration_days is None else duration_days)
    duration = max(duration, 0.0)
    beta_gp = max(float(params["beta_GP"]), 0.0)
    return beta_gp * (1.0 - np.exp(-beta_gp_hydrolysis_rate_constant(params) * duration))


def beta_gp_cycle_phosphate(t: float, params: Dict[str, float]) -> float:
    cycle_start = float(params.get("_cycle_start", 0.0))
    cycle_time_value = max(0.0, float(t) - cycle_start)
    beta_gp = max(float(params.get("beta_GP_cycle", params["beta_GP"])), 0.0)
    k_hyd = beta_gp_hydrolysis_rate_constant(params)
    return beta_gp * (1.0 - np.exp(-k_hyd * cycle_time_value))


def phosphate_target(t: float, params: Dict[str, float]) -> float:
    return params["P_ext"] + beta_gp_cycle_phosphate(t, params)


def beta_gp_cycle_source(t: float, params: Dict[str, float]) -> float:
    cycle_start = float(params.get("_cycle_start", 0.0))
    cycle_time_value = max(0.0, float(t) - cycle_start)
    beta_gp_remaining = max(float(params.get("beta_GP_cycle", params["beta_GP"])), 0.0) * np.exp(
        -beta_gp_hydrolysis_rate_constant(params) * cycle_time_value
    )
    return phosphate_release_rate(beta_gp_remaining, beta_gp_hydrolysis_rate_constant(params))


def cycle_time(t: float, params: Dict[str, float]) -> float:
    cycle_start = float(params.get("_cycle_start", 0.0))
    return max(0.0, t - cycle_start)


def _validated_nonnegative(name: str, value: float, tolerance: float = NEGATIVE_STATE_TOLERANCE) -> float:
    value = float(value)
    if value < -tolerance:
        raise ValueError(f"{name} is negative beyond tolerance: {value:.6g}")
    return max(value, 0.0)


def _validate_state_values(C: float, P: float, B: float, A: float, H: float) -> tuple[float, float, float, float, float]:
    return (
        _validated_nonnegative("C", C),
        _validated_nonnegative("P", P),
        _validated_nonnegative("B", B),
        _validated_nonnegative("A", A),
        _validated_nonnegative("H", H),
    )


def _project_nonnegative_state_array(name: str, values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains nonfinite values.")
    min_value = float(np.min(arr))
    if min_value < -POSTPROCESS_NEGATIVE_STATE_TOLERANCE:
        raise ValueError(f"{name} has values below -{POSTPROCESS_NEGATIVE_STATE_TOLERANCE:g}: min={min_value:.6g}")
    return np.maximum(arr, 0.0)


def _project_nonnegative_state_matrix(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("State trajectory contains nonfinite values.")
    min_value = float(np.min(arr))
    if min_value < -POSTPROCESS_NEGATIVE_STATE_TOLERANCE:
        raise ValueError(
            f"State trajectory has values below -{POSTPROCESS_NEGATIVE_STATE_TOLERANCE:g}: min={min_value:.6g}"
        )
    return np.maximum(arr, 0.0)


def supersaturation(C: float, P: float, params: Dict[str, float]) -> float:
    C = _validated_nonnegative("C", C)
    P = _validated_nonnegative("P", P)
    K_A = max(params["K_A"], MIN_K_A)
    s = (C ** params["alpha"] * P ** params["beta"]) / K_A
    return _validated_nonnegative("S_A", s)


def mv_activity(t: float, params: Dict[str, float]) -> float:
    del t, params
    return 1.0


def precursor_formation(t: float, C: float, P: float, A: float, params: Dict[str, float], H: float = 0.0) -> float:
    del t
    S_A = supersaturation(C, P, params)
    if S_A <= 1.0:
        return 0.0

    k_f = float(params["k_f"])
    capacity = max(0.0, 1.0 - (max(float(A), 0.0) + max(float(H), 0.0)) / max(float(params["M_max"]), MIN_DENOMINATOR))
    return k_f * (S_A - 1.0) ** params["n"] * capacity


def ascorbate_concentration(params: Dict[str, float], elapsed_days: float = 0.0) -> float:
    a_aa0_cycle = _validated_nonnegative("a_AA0", float(params["a_AA0"]))
    half_life_days = max(float(params.get("t_AA_half_life", 24.0)) / 24.0, MIN_DENOMINATOR)
    elapsed = max(float(elapsed_days), 0.0)
    return float(a_aa0_cycle * np.exp(-np.log(2.0) * elapsed / half_life_days))


def matrix_support_factor(params: Dict[str, float], elapsed_days: float = 0.0) -> float:
    a_aa = ascorbate_concentration(params, elapsed_days)
    a_aa0 = _validated_nonnegative("a_AA0", float(params["a_AA0"]))
    k_mat_aa = max(float(params.get("K_mat_AA", ASCORBATE_HALF_SUPPORT)), MIN_DENOMINATOR)
    support = a_aa / (k_mat_aa + a_aa) if a_aa > 0.0 else 0.0
    baseline_support = a_aa0 / (k_mat_aa + a_aa0) if a_aa0 > 0.0 else 0.0
    return support / max(baseline_support, MIN_DENOMINATOR)


def matrix_support_factor_at_time(t: float, params: Dict[str, float]) -> float:
    return matrix_support_factor(params, cycle_time(t, params))


def maturation(C: float, P: float, A: float, H: float, params: Dict[str, float], t: float = 0.0) -> float:
    C, P, _, A, H = _validate_state_values(C, P, 0.0, A, H)
    k_m = float(params["k_m"])
    f_mat = matrix_support_factor_at_time(t, params)
    return k_m * A * f_mat


def mineralization_rhs(t: float, y: np.ndarray, params: Dict[str, float]) -> List[float]:
    C, P, B, A, H = y
    # Adaptive solvers may evaluate rejected trial stages slightly outside the
    # physical domain. Do not mutate accepted states here; validate the returned
    # trajectory after integration.
    C = max(float(C), 0.0)
    P = max(float(P), 0.0)
    B = max(float(B), 0.0)
    A = max(float(A), 0.0)
    H = max(float(H), 0.0)

    ion_activity = C ** params["alpha"] * P ** params["beta"]
    S_A = ion_activity / params["_K_A"]
    capacity = max(0.0, 1.0 - (A + H) / params["_M_max"])
    Rf = params["k_f"] * (S_A - 1.0) ** params["n"] * capacity if S_A > 1.0 else 0.0
    Rm = params["k_m"] * A * matrix_support_factor_at_time(t, params)
    J_bgp = params["k_hyd"] * B
    extra_C_maturation = params["r_H"] - params["r_A"]

    dCdt = -params["r_A"] * Rf - extra_C_maturation * Rm
    dPdt = J_bgp - Rf
    dBdt = -J_bgp
    dAdt = Rf - Rm
    dHdt = Rm
    # Boundary fluxes are disabled as complete reactions, preserving both
    # elemental balances instead of clipping individual derivatives.
    if C <= MIN_CONCENTRATION:
        Rf = 0.0
        Rm = 0.0
    if P <= MIN_CONCENTRATION:
        Rf = 0.0
    if B <= MIN_CONCENTRATION:
        J_bgp = 0.0
    if A <= MIN_CONCENTRATION:
        Rm = 0.0
    dCdt = -params["r_A"] * Rf - extra_C_maturation * Rm
    dPdt = J_bgp - Rf
    dBdt = -J_bgp
    dAdt = Rf - Rm
    dHdt = Rm
    return [dCdt, dPdt, dBdt, dAdt, dHdt]


def simulate(
    params: Dict[str, float],
    t_end: float,
    dt: float,
    media_interval: float,
) -> SimulationResult:
    params = dict(params)
    params["media_interval"] = float(media_interval)
    params["media_renewal_fraction"] = min(max(float(params.get("media_renewal_fraction", 1.0)), 0.0), 1.0)
    params.setdefault("eta_48h", ETA_48H)
    synchronize_hydrolysis_parameters(params, dt, media_interval)
    params["k_f"] = float(params["k_f"])
    params["M_max"] = max(float(params["M_max"]), MIN_DENOMINATOR)
    params["_K_A"] = max(float(params["K_A"]), MIN_K_A)
    params["_M_max"] = params["M_max"]
    params["r_H"] = float(max(params["r_H"], params["r_A"]))
    media_days = list(np.arange(media_interval, t_end, media_interval))
    breakpoints = media_days + [t_end]

    y = np.array(
        [
            max(float(params["C_ext"]), 0.0),
            max(float(params["P_ext"]), 0.0),
            max(float(params["beta_GP"]), 0.0),
            0.0,
            0.0,
        ],
        dtype=float,
    )
    t0 = 0.0
    times_all: List[np.ndarray] = []
    states_all: List[np.ndarray] = []
    reset_status_all: List[np.ndarray] = []
    cycle_index_all: List[np.ndarray] = []
    cycle_start_all: List[np.ndarray] = []
    time_since_reset_all: List[np.ndarray] = []

    for cycle_idx, t1 in enumerate(breakpoints):
        params["_cycle_start"] = t0
        params["_cycle_length"] = t1 - t0
        n_steps = max(2, int(np.ceil((t1 - t0) / dt)) + 1)
        t_eval = np.linspace(t0, t1, n_steps)
        sol = solve_ivp(
            mineralization_rhs,
            (t0, t1),
            y,
            args=(params,),
            t_eval=t_eval,
            method=str(params["solver_method"]),
            rtol=float(params["solver_rtol"]),
            atol=float(params["solver_atol"]),
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
        y_segment = _project_nonnegative_state_matrix(y_segment)
        status = np.full(t_segment.shape, "continuous", dtype=object)
        if np.isclose(t1, media_days).any() and status.size:
            status[-1] = "pre_reset"
        times_all.append(t_segment)
        states_all.append(y_segment)
        reset_status_all.append(status)
        cycle_index_all.append(np.full(t_segment.shape, cycle_idx, dtype=int))
        cycle_start_all.append(np.full(t_segment.shape, t0, dtype=float))
        time_since_reset_all.append(t_segment - t0)

        y = _project_nonnegative_state_matrix(sol.y[:, -1].reshape(5, 1))[:, 0].copy()

        if np.isclose(t1, media_days).any():
            y_reset = y.copy()
            renew = float(params["media_renewal_fraction"])
            y_reset[0] = (1.0 - renew) * max(float(y[0]), 0.0) + renew * max(float(params["C_ext"]), 0.0)
            y_reset[1] = (1.0 - renew) * max(float(y[1]), 0.0) + renew * max(float(params["P_ext"]), 0.0)
            y_reset[2] = (1.0 - renew) * max(float(y[2]), 0.0) + renew * max(float(params["beta_GP"]), 0.0)
            times_all.append(np.array([t1]))
            states_all.append(y_reset.reshape(5, 1))
            reset_status_all.append(np.array(["post_reset"], dtype=object))
            cycle_index_all.append(np.array([cycle_idx + 1], dtype=int))
            cycle_start_all.append(np.array([t1], dtype=float))
            time_since_reset_all.append(np.array([0.0], dtype=float))
            y = y_reset

        t0 = t1

    t = np.concatenate(times_all)
    Y = np.concatenate(states_all, axis=1)
    C, P, B, A, H = Y
    C = _project_nonnegative_state_array("C", C)
    P = _project_nonnegative_state_array("P", P)
    B = _project_nonnegative_state_array("B", B)
    A = _project_nonnegative_state_array("A", A)
    H = _project_nonnegative_state_array("H", H)
    reset_status = np.concatenate(reset_status_all)
    cycle_index = np.concatenate(cycle_index_all)
    cycle_start = np.concatenate(cycle_start_all)
    time_since_reset = np.concatenate(time_since_reset_all)

    ion_activity = C ** float(params["alpha"]) * P ** float(params["beta"])
    S_A = np.maximum(ion_activity / params["_K_A"], 0.0)
    J_bgp = params["k_hyd"] * B
    capacity = np.maximum(1.0 - (A + H) / params["_M_max"], 0.0)
    Rf = params["k_f"] * np.maximum(S_A - 1.0, 0.0) ** float(params["n"]) * capacity
    f_MV = np.ones(len(t), dtype=float)
    f_mat = np.array([matrix_support_factor(params, elapsed) for elapsed in time_since_reset], dtype=float)
    Rm = float(params["k_m"]) * A * f_mat
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
        f_MV=f_MV,
        f_mat=f_mat,
        media_days=media_days,
        reset_status=reset_status,
        cycle_index=cycle_index,
        cycle_start=cycle_start,
        time_since_reset=time_since_reset,
    )
    validate_simulation_state(sim)
    return sim


def cycle_aware_trapezoid(y: np.ndarray, t: np.ndarray, cycle_index: np.ndarray) -> float:
    total = 0.0
    for cycle in np.unique(cycle_index):
        mask = cycle_index == cycle
        if np.count_nonzero(mask) > 1:
            total += float(np.trapezoid(np.asarray(y)[mask], np.asarray(t)[mask]))
    return total


def first_threshold_crossing_time(result: SimulationResult) -> float:
    for cycle in np.unique(result.cycle_index):
        idx = np.where(result.cycle_index == cycle)[0]
        idx = idx[result.reset_status[idx] != "post_reset"]
        s = result.S_A[idx]
        t = result.t[idx]
        above = np.where(s >= 1.0)[0]
        if above.size == 0:
            continue
        j = int(above[0])
        if j == 0:
            return float(t[j])
        s0, s1 = float(s[j - 1]), float(s[j])
        t0, t1 = float(t[j - 1]), float(t[j])
        if np.isclose(s1, s0):
            return t1
        return float(t0 + (1.0 - s0) * (t1 - t0) / (s1 - s0))
    return float("nan")


def analytical_baseline_quantities(params: Dict[str, float], media_interval: float) -> Dict[str, float]:
    params = dict(params)
    params["media_interval"] = float(media_interval)
    q_pi = beta_gp_available_phosphate(params, media_interval)
    c_ext = max(float(params["C_ext"]), MIN_DENOMINATOR)
    p_crit = (float(params["K_A"]) / c_ext ** float(params["alpha"])) ** (1.0 / float(params["beta"]))
    q_pi_crit = p_crit - float(params["P_ext"])
    k_hyd = beta_gp_hydrolysis_rate_constant(params)
    beta_gp = max(float(params["beta_GP"]), 0.0)
    j_beta_gp_initial = phosphate_release_rate(beta_gp, k_hyd)
    if beta_gp > MIN_DENOMINATOR and 0.0 < q_pi_crit < beta_gp:
        t_onset = -np.log1p(-q_pi_crit / beta_gp) / k_hyd if k_hyd > MIN_DENOMINATOR else float("nan")
    elif q_pi_crit <= 0.0:
        t_onset = 0.0
    else:
        t_onset = float("nan")
    return {
        "q_Pi": float(q_pi),
        "P_crit": float(p_crit),
        "q_Pi_crit": float(q_pi_crit),
        "J_betaGP": float(j_beta_gp_initial),
        "k_hyd": float(k_hyd),
        "t_onset": float(t_onset),
    }


def validate_simulation_state(result: SimulationResult, tolerance: float = NEGATIVE_STATE_TOLERANCE) -> None:
    arrays = {
        "C": result.C,
        "P": result.P,
        "B": result.B,
        "A": result.A,
        "H": result.H,
        "J_bgp": result.J_bgp,
        "R_f": result.Rf,
        "R_m": result.Rm,
    }
    for name, values in arrays.items():
        arr = np.asarray(values, dtype=float)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains nonfinite values.")
        min_value = float(np.min(arr))
        if min_value < -tolerance:
            raise ValueError(f"{name} has values below -{tolerance:g}: min={min_value:.6g}")


def regime_metrics(result: SimulationResult) -> Dict[str, float | str]:
    eps = MIN_DENOMINATOR
    auc_a = cycle_aware_trapezoid(result.A, result.t, result.cycle_index)
    auc_h = cycle_aware_trapezoid(result.H, result.t, result.cycle_index)
    int_rf = cycle_aware_trapezoid(result.Rf, result.t, result.cycle_index)
    int_rm = cycle_aware_trapezoid(result.Rm, result.t, result.cycle_index)
    fraction_supersaturated = float(np.mean(result.S_A > 1.0))

    onset_time = first_threshold_crossing_time(result)
    h_threshold = 0.05 * float(np.max(result.H))
    h_idx = np.where(result.H >= h_threshold)[0]
    h_onset = float(result.t[h_idx[0]]) if h_threshold > 0.0 and len(h_idx) else float("nan")

    rf_mean = float(np.mean(result.Rf))
    pulse_index = float(np.std(result.Rf) / max(rf_mean, eps)) if rf_mean > eps else 0.0
    integrated_maturation_fraction = int_rm / max(int_rf, eps)

    if fraction_supersaturated < 0.25:
        regime = "phosphate_or_supersaturation_limited"
    elif int_rf <= eps:
        regime = "nucleation_limited"
    elif integrated_maturation_fraction < 0.5 and auc_a > auc_h:
        regime = "maturation_limited"
    elif pulse_index > 1.0:
        regime = "pulse_driven"
    else:
        regime = "apatite_growth"

    return {
        "fraction_time_supersaturated": fraction_supersaturated,
        "onset_time_SA_days": onset_time,
        "onset_time_H_days": h_onset,
        "AUC_A": auc_a,
        "AUC_H": auc_h,
        "integral_Rf": int_rf,
        "integral_Rm": int_rm,
        "maturation_ratio": integrated_maturation_fraction,
        "integrated_maturation_fraction": integrated_maturation_fraction,
        "pulse_index": pulse_index,
        "regime": regime,
    }


def transient_metrics(result: SimulationResult) -> Dict[str, float]:
    peak_idx = int(np.argmax(result.A))
    A_peak = float(result.A[peak_idx])
    t_peak = float(result.t[peak_idx])
    A_final = float(result.A[-1])
    H_final = float(result.H[-1])
    phi_H_final = H_final / (A_final + H_final) if A_final + H_final > MIN_DENOMINATOR else float("nan")
    post_peak_A = result.A[peak_idx:]
    A_post_peak_min = float(np.min(post_peak_A)) if len(post_peak_A) else A_final
    decline_fraction = float((A_peak - A_post_peak_min) / A_peak) if A_peak > 0 else np.nan
    H_after_peak = float(result.H[-1] - result.H[peak_idx])
    metrics = {
        "A_peak": A_peak,
        "t_peak_days": t_peak,
        "global_A_peak": A_peak,
        "global_A_peak_time": t_peak,
        "global_A_peak_time_endpoint_dependent": t_peak,
        "A_final": A_final,
        "phi_H_final": phi_H_final,
        "A_post_peak_min": A_post_peak_min,
        "decline_fraction": decline_fraction,
        "H_final": H_final,
        "H_after_peak": H_after_peak,
    }
    metrics.update(regime_metrics(result))
    return metrics
