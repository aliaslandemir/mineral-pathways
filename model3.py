import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# 1. Parameters
# ============================================================

p = {
    "C_ext": 1.2,
    "P_ext": 1.0,

    "k_C": 0.7,
    "k_P": 0.7,

    "r_A": 1.35,
    "r_H": 1.63,

    "alpha": 3.0,
    "beta": 2.0,
    "K_A": 1.4,

    "N_v": 1.0,
    "k_f": 0.05,
    "n": 2.0,

    "k_m": 0.10,
    "K_Cm": 0.5,
    "phi_disp": 1.0,

    "Mg": 0.0,
    "Cit": 0.0,
    "lambda_Mg": 2.0,
    "lambda_Cit": 3.0,

    "Phi_f": 1.0,
    "k_d": 0.03,
    "k_out": 0.01,
}


sweep = {
    # ion supply
    "C_ext": [1.1, 1.2, 1.3],
    "P_ext": [0.8, 1.0, 1.2, 1.5],

    # transport / replenishment
    "k_C": [0.3, 0.7, 1.5, 3.0],
    "k_P": [0.3, 0.7, 1.5, 3.0],

    # phase composition
    "r_A": [1.25, 1.35, 1.45],
    "r_H": [1.58, 1.63, 1.67],

    # supersaturation proxy
    "K_A": [1.1, 1.2, 1.4, 1.6],
    "alpha": [2.0, 3.0, 4.0],
    "beta": [1.0, 2.0, 3.0],

    # precursor formation
    "N_v": [0.25, 0.5, 1.0, 2.0, 4.0],
    "k_f": [0.01, 0.03, 0.05, 0.10, 0.20],
    "n": [1.0, 2.0, 3.0],

    # maturation
    "k_m": [0.03, 0.05, 0.10, 0.25, 0.50],
    "K_Cm": [0.2, 0.5, 1.0],
    "phi_disp": [0.25, 0.5, 1.0, 2.0],

    # stabilization
    "Mg": [0.0, 0.1, 0.5, 1.0],
    "Cit": [0.0, 0.05, 0.1, 0.5],
    "lambda_Mg": [0.5, 1.0, 2.0, 5.0],
    "lambda_Cit": [0.5, 1.0, 3.0, 10.0],

    # precursor loss
    "k_d": [0.005, 0.01, 0.03, 0.10],
    "k_out": [0.0, 0.005, 0.01, 0.05],
}


# ============================================================
# 2. Rate laws
# ============================================================


def supersaturation(C, P, p):
    C = max(C, 0.0)
    P = max(P, 0.0)
    return (C ** p["alpha"] * P ** p["beta"]) / p["K_A"]


def precursor_formation(C, P, p):
    S_A = supersaturation(C, P, p)
    if S_A <= 1.0:
        return 0.0
    return p["N_v"] * p["k_f"] * p["Phi_f"] * (S_A - 1.0) ** p["n"]


def maturation(C, A, p):
    C = max(C, 0.0)
    A = max(A, 0.0)
    calcium_factor = C / (C + p["K_Cm"])
    mg_factor = 1.0 / (1.0 + p["lambda_Mg"] * p["Mg"])
    cit_factor = 1.0 / (1.0 + p["lambda_Cit"] * p["Cit"])
    return p["k_m"] * A * p["phi_disp"] * calcium_factor * mg_factor * cit_factor


def precursor_dissolution(A, p):
    A = max(A, 0.0)
    return p["k_d"] * A


def precursor_removal(A, p):
    A = max(A, 0.0)
    return p["k_out"] * A


# ============================================================
# 3. ODE system
# ============================================================


def mineralization_model(t, y, p):
    C, P, A, H = y

    C = max(C, 0.0)
    P = max(P, 0.0)
    A = max(A, 0.0)

    R_f = precursor_formation(C, P, p)
    R_m = maturation(C, A, p)
    R_d = precursor_dissolution(A, p)
    R_out = precursor_removal(A, p)

    dCdt = (
        p["k_C"] * (p["C_ext"] - C)
        - p["r_A"] * R_f
        - (p["r_H"] - p["r_A"]) * R_m
        + p["r_A"] * R_d
    )

    dPdt = p["k_P"] * (p["P_ext"] - P) - R_f + R_d
    dAdt = R_f - R_m - R_d - R_out
    dHdt = R_m

    return [dCdt, dPdt, dAdt, dHdt]


# ============================================================
# 4. Simulation with medium replacement
# ============================================================


def simulate_with_media_change(p, t_end=30, dt=0.02, media_interval=2):
    media_days = list(np.arange(media_interval, t_end, media_interval))
    breakpoints = media_days + [t_end]

    y = np.array([p["C_ext"], p["P_ext"], 0.0, 0.0])

    t0 = 0.0
    times_all = []
    states_all = []

    for t1 in breakpoints:
        n_steps = max(2, int(np.ceil((t1 - t0) / dt)) + 1)
        t_eval = np.linspace(t0, t1, n_steps)

        sol = solve_ivp(
            mineralization_model,
            (t0, t1),
            y,
            args=(p,),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-6,
            atol=1e-9,
        )

        if not sol.success:
            raise RuntimeError(sol.message)

        if times_all:
            times_all.append(sol.t[1:])
            states_all.append(sol.y[:, 1:])
        else:
            times_all.append(sol.t)
            states_all.append(sol.y)

        y = sol.y[:, -1].copy()

        # Medium replacement: reset soluble ions only
        if np.isclose(t1, media_days).any():
            y[0] = p["C_ext"]
            y[1] = p["P_ext"]

        t0 = t1

    t = np.concatenate(times_all)
    Y = np.concatenate(states_all, axis=1)

    return t, Y, media_days


# ============================================================
# 5. Metrics and rates
# ============================================================


def mineral_metrics(t, A, H):
    A_peak = np.max(A)
    t_A_peak = t[np.argmax(A)]
    H_final = H[-1]

    if H_final > 0:
        onset_idx = np.where(H >= 0.1 * H_final)[0]
        t_H_onset = t[onset_idx[0]] if len(onset_idx) > 0 else np.nan
    else:
        t_H_onset = np.nan

    return {
        "ACP peak": A_peak,
        "ACP peak time [days]": t_A_peak,
        "Final mature mineral H": H_final,
        "H onset time, 10% final [days]": t_H_onset,
    }


def rate_arrays(t, Y, p):
    C, P, A, H = Y

    Rf = np.array([precursor_formation(C[i], P[i], p) for i in range(len(t))])
    Rm = np.array([maturation(C[i], A[i], p) for i in range(len(t))])
    Rd = np.array([precursor_dissolution(A[i], p) for i in range(len(t))])
    Rout = np.array([precursor_removal(A[i], p) for i in range(len(t))])

    return Rf, Rm, Rd, Rout


def summarize_run(t, Y):
    C, P, A, H = Y
    metrics = mineral_metrics(t, A, H)
    return {
        "H_final": metrics["Final mature mineral H"],
        "A_peak": metrics["ACP peak"],
        "t_H_onset": metrics["H onset time, 10% final [days]"],
        "A_final": A[-1],
        "C_final": C[-1],
        "P_final": P[-1],
    }


# ============================================================
# 6. Plot helpers
# ============================================================


def add_media_lines(ax, media_days):
    for d in media_days:
        ax.axvline(d, linestyle="--", alpha=0.20, linewidth=1.0, color="#6b7280")


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


def _norm_change_from_baseline(values, baseline):
    baseline = max(abs(baseline), 1e-12)
    return [(v - baseline) / baseline for v in values]


def _local_log_sensitivity(xs, ys, x0, y0):
    pairs = sorted(zip(xs, ys), key=lambda t: t[0])
    lower = None
    upper = None
    for x, y in pairs:
        if x < x0:
            lower = (x, y)
        elif x > x0 and upper is None:
            upper = (x, y)
    if lower is None or upper is None:
        return np.nan
    x1, y1 = lower
    x2, y2 = upper
    if x1 <= 0 or x2 <= 0 or y1 <= 0 or y2 <= 0 or x0 <= 0 or y0 <= 0:
        return np.nan
    return (np.log(y2) - np.log(y1)) / (np.log(x2) - np.log(x1))


def run_sensitivity_sweep(base_params, sweep_dict, t_end=30, media_interval=2):
    t_base, Y_base, _ = simulate_with_media_change(
        base_params,
        t_end=t_end,
        media_interval=media_interval,
    )
    base = summarize_run(t_base, Y_base)

    results = {}
    for param, values in sweep_dict.items():
        rows = []
        for value in values:
            p_temp = base_params.copy()
            p_temp[param] = value
            t, Y, _ = simulate_with_media_change(
                p_temp,
                t_end=t_end,
                media_interval=media_interval,
            )
            s = summarize_run(t, Y)
            rows.append({
                "value": value,
                "H_final": s["H_final"],
                "A_peak": s["A_peak"],
                "t_H_onset": s["t_H_onset"],
            })

        results[param] = rows

    return base, results


def save_sensitivity_table(path, base, results):
    with open(path, "w", encoding="utf-8") as f:
        f.write("parameter,value,H_final,A_peak,t_H_onset,dH_final_rel,dA_peak_rel\n")
        for param, rows in results.items():
            for row in rows:
                dH = (row["H_final"] - base["H_final"]) / max(abs(base["H_final"]), 1e-12)
                dA = (row["A_peak"] - base["A_peak"]) / max(abs(base["A_peak"]), 1e-12)
                f.write(
                    f"{param},{row['value']},{row['H_final']:.8g},{row['A_peak']:.8g},{row['t_H_onset']:.8g},{dH:.8g},{dA:.8g}\n"
                )


def save_sensitivity_indices(path, base_params, base, results):
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "parameter,local_logS_H_final,local_logS_A_peak,local_logS_t_H_onset,"
            "global_span_H_final,global_span_A_peak,global_span_t_H_onset\n"
        )
        for param, rows in results.items():
            xs = [r["value"] for r in rows]
            h = [r["H_final"] for r in rows]
            a = [r["A_peak"] for r in rows]
            t0 = [r["t_H_onset"] for r in rows]

            s_h = _local_log_sensitivity(xs, h, base_params[param], base["H_final"])
            s_a = _local_log_sensitivity(xs, a, base_params[param], base["A_peak"])
            s_t = _local_log_sensitivity(xs, t0, base_params[param], base["t_H_onset"])

            span_h = np.nanmax(_norm_change_from_baseline(h, base["H_final"])) - np.nanmin(
                _norm_change_from_baseline(h, base["H_final"])
            )
            span_a = np.nanmax(_norm_change_from_baseline(a, base["A_peak"])) - np.nanmin(
                _norm_change_from_baseline(a, base["A_peak"])
            )
            span_t = np.nanmax(_norm_change_from_baseline(t0, base["t_H_onset"])) - np.nanmin(
                _norm_change_from_baseline(t0, base["t_H_onset"])
            )

            f.write(
                f"{param},{s_h:.8g},{s_a:.8g},{s_t:.8g},{span_h:.8g},{span_a:.8g},{span_t:.8g}\n"
            )


def plot_sensitivity_profiles(results, base, out_dir):
    params = list(results.keys())
    n = len(params)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), sharey=False)
    axes = np.array(axes).reshape(-1)

    for i, param in enumerate(params):
        ax = axes[i]
        values = [r["value"] for r in results[param]]
        hvals = [r["H_final"] for r in results[param]]
        avals = [r["A_peak"] for r in results[param]]

        dH = _norm_change_from_baseline(hvals, base["H_final"])
        dA = _norm_change_from_baseline(avals, base["A_peak"])

        ax.plot(values, dH, marker="o", linewidth=2.0, color="#0f766e", label="Final H")
        ax.plot(values, dA, marker="s", linewidth=1.8, color="#b45309", label="ACP peak")
        ax.axhline(0.0, color="#6b7280", linewidth=0.8)
        ax.set_title(param)
        ax.set_xlabel(param)
        ax.grid(alpha=0.2)

        if i % ncols == 0:
            ax.set_ylabel("Relative change vs baseline")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
    fig.suptitle("One-at-a-time sensitivity profiles", y=1.02)
    save_figure(fig, os.path.join(out_dir, "fig_sensitivity_profiles.pdf"))


def plot_sensitivity_tornado(results, base, out_dir):
    params = list(results.keys())
    span_h = []
    span_a = []

    for param in params:
        rows = results[param]
        hvals = [r["H_final"] for r in rows]
        avals = [r["A_peak"] for r in rows]
        dH = _norm_change_from_baseline(hvals, base["H_final"])
        dA = _norm_change_from_baseline(avals, base["A_peak"])
        span_h.append(max(dH) - min(dH))
        span_a.append(max(dA) - min(dA))

    score = np.array(span_h) + np.array(span_a)
    order = np.argsort(score)[::-1]
    params_sorted = [params[i] for i in order]
    span_h_sorted = [span_h[i] for i in order]
    span_a_sorted = [span_a[i] for i in order]

    fig, ax = plt.subplots(figsize=(9.0, 8.4))
    y = np.arange(len(params_sorted))
    ax.barh(y - 0.18, span_h_sorted, height=0.34, color="#1d4ed8", label="Final H span")
    ax.barh(y + 0.18, span_a_sorted, height=0.34, color="#b45309", label="ACP peak span")
    ax.set_yticks(y)
    ax.set_yticklabels(params_sorted)
    ax.invert_yaxis()
    ax.set_xlabel("Global OAT sensitivity span (max-min relative change)")
    ax.set_title("Parameter influence ranking by endpoint sensitivity")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=True)

    save_figure(fig, os.path.join(out_dir, "fig_sensitivity_tornado.pdf"))


def plot_sensitivity_heatmap(results, base, out_dir):
    params = list(results.keys())
    max_len = max(len(rows) for rows in results.values())
    mat = np.full((len(params), max_len), np.nan)

    for i, param in enumerate(params):
        rows = results[param]
        hvals = [r["H_final"] for r in rows]
        dH = _norm_change_from_baseline(hvals, base["H_final"])
        mat[i, :len(dH)] = dH

    fig, ax = plt.subplots(figsize=(12, 8.0))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-np.nanmax(np.abs(mat)), vmax=np.nanmax(np.abs(mat)))
    ax.set_yticks(np.arange(len(params)))
    ax.set_yticklabels(params)
    ax.set_xticks(np.arange(max_len))
    ax.set_xticklabels([str(i + 1) for i in range(max_len)])
    ax.set_xlabel("Sweep point index")
    ax.set_title("Relative Final H change across one-at-a-time sweep points")
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("(H_final - baseline) / baseline")

    save_figure(fig, os.path.join(out_dir, "fig_sensitivity_heatmap.pdf"))


def plot_local_log_sensitivity(results, base, base_params, out_dir):
    params = []
    sh = []
    sa = []
    st = []
    for param, rows in results.items():
        xs = [r["value"] for r in rows]
        h = [r["H_final"] for r in rows]
        a = [r["A_peak"] for r in rows]
        t0 = [r["t_H_onset"] for r in rows]
        params.append(param)
        sh.append(_local_log_sensitivity(xs, h, base_params[param], base["H_final"]))
        sa.append(_local_log_sensitivity(xs, a, base_params[param], base["A_peak"]))
        st.append(_local_log_sensitivity(xs, t0, base_params[param], base["t_H_onset"]))

    score = np.nan_to_num(np.abs(sh), nan=0.0) + np.nan_to_num(np.abs(sa), nan=0.0)
    order = np.argsort(score)[::-1]
    params = [params[i] for i in order]
    sh = [sh[i] for i in order]
    sa = [sa[i] for i in order]
    st = [st[i] for i in order]

    fig, ax = plt.subplots(figsize=(9.2, 8.4))
    y = np.arange(len(params))
    ax.barh(y - 0.25, sh, height=0.24, color="#0f766e", label="log-sensitivity: Final H")
    ax.barh(y, sa, height=0.24, color="#c2410c", label="log-sensitivity: ACP peak")
    ax.barh(y + 0.25, st, height=0.24, color="#6d28d9", label="log-sensitivity: H onset time")
    ax.axvline(0.0, color="#6b7280", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(params)
    ax.invert_yaxis()
    ax.set_xlabel("Local log-sensitivity around baseline")
    ax.set_title("Local elasticity indices by parameter")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=True)

    save_figure(fig, os.path.join(out_dir, "fig_local_log_sensitivity.pdf"))


# ============================================================
# 7. Main analysis
# ============================================================

if __name__ == "__main__":
    out_dir = "results/sweepsent"
    os.makedirs(out_dir, exist_ok=True)

    t_end = 30
    media_interval = 2

    # Baseline
    t, Y, media_days = simulate_with_media_change(
        p,
        t_end=t_end,
        media_interval=media_interval,
    )

    C, P, A, H = Y
    Rf, Rm, Rd, Rout = rate_arrays(t, Y, p)

    # Professional baseline figure set
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 8.5), sharex=True)

    ax = axes[0]
    ax.plot(t, C, label="Calcium C", color="#2563eb", linewidth=2.0)
    ax.plot(t, P, label="Phosphate P", color="#0f766e", linewidth=2.0)
    add_media_lines(ax, media_days)
    ax.set_ylabel("Ion concentration [mM]")
    ax.legend(frameon=True)
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(t, A, label="ACP precursor A", color="#b45309", linewidth=2.0)
    ax.plot(t, H, label="Apatite-like mineral H", color="#7c3aed", linewidth=2.0)
    add_media_lines(ax, media_days)
    ax.set_ylabel("Mineral amount [P-equiv]")
    ax.legend(frameon=True)
    ax.grid(alpha=0.2)

    ax = axes[2]
    ax.plot(t, Rf, label="Precursor formation R_f", linewidth=1.8)
    ax.plot(t, Rm, label="Maturation R_m", linewidth=1.8)
    ax.plot(t, Rd, label="Precursor dissolution R_d", linewidth=1.8)
    ax.plot(t, Rout, label="Precursor removal R_out", linewidth=1.8)
    add_media_lines(ax, media_days)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Rate [P-equiv/day]")
    ax.legend(frameon=True, ncol=2)
    ax.grid(alpha=0.2)

    save_figure(fig, os.path.join(out_dir, "fig_baseline_dynamics_model3.pdf"))

    # Sensitivity analysis
    base_summary, sens_results = run_sensitivity_sweep(
        p,
        sweep,
        t_end=t_end,
        media_interval=media_interval,
    )

    save_sensitivity_table(
        os.path.join(out_dir, "sensitivity_summary.csv"),
        base_summary,
        sens_results,
    )
    save_sensitivity_indices(
        os.path.join(out_dir, "sensitivity_indices.csv"),
        p,
        base_summary,
        sens_results,
    )

    plot_sensitivity_profiles(sens_results, base_summary, out_dir)
    plot_sensitivity_tornado(sens_results, base_summary, out_dir)
    plot_sensitivity_heatmap(sens_results, base_summary, out_dir)
    plot_local_log_sensitivity(sens_results, base_summary, p, out_dir)

    print("\nSaved model3 outputs to: results/sweepsent/")
