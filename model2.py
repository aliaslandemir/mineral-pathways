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
    "K_A": 1.4,   # tune so S_A baseline is mildly > 1

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

    return (
        p["N_v"]
        * p["k_f"]
        * p["Phi_f"]
        * (S_A - 1.0) ** p["n"]
    )


def maturation(C, A, p):
    C = max(C, 0.0)
    A = max(A, 0.0)

    calcium_factor = C / (C + p["K_Cm"])
    mg_factor = 1.0 / (1.0 + p["lambda_Mg"] * p["Mg"])
    cit_factor = 1.0 / (1.0 + p["lambda_Cit"] * p["Cit"])

    return (
        p["k_m"]
        * A
        * p["phi_disp"]
        * calcium_factor
        * mg_factor
        * cit_factor
    )


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

    dPdt = (
        p["k_P"] * (p["P_ext"] - P)
        - R_f
        + R_d
    )

    dAdt = R_f - R_m - R_d - R_out
    dHdt = R_m

    return [dCdt, dPdt, dAdt, dHdt]


# ============================================================
# 4. Simulation with medium replacement
# ============================================================

def simulate_with_media_change(p, t_end=30, dt=0.02, media_interval=2):
    media_days = list(np.arange(media_interval, t_end, media_interval))
    breakpoints = media_days + [t_end]

    y = np.array([
        p["C_ext"],
        p["P_ext"],
        0.0,
        0.0,
    ])

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


def print_metrics(label, t, Y):
    C, P, A, H = Y

    print("\n" + label)
    print("-" * len(label))

    for key, value in mineral_metrics(t, A, H).items():
        print(f"{key}: {value:.4f}")


def rate_arrays(t, Y, p):
    C, P, A, H = Y

    Rf = np.array([precursor_formation(C[i], P[i], p) for i in range(len(t))])
    Rm = np.array([maturation(C[i], A[i], p) for i in range(len(t))])
    Rd = np.array([precursor_dissolution(A[i], p) for i in range(len(t))])
    Rout = np.array([precursor_removal(A[i], p) for i in range(len(t))])

    return Rf, Rm, Rd, Rout


# ============================================================
# 6. Plot helpers
# ============================================================

def add_media_lines(ax, media_days):
    for d in media_days:
        ax.axvline(d, linestyle="--", alpha=0.25, linewidth=1.2)


def save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


# ============================================================
# 7. Main analysis
# ============================================================

if __name__ == "__main__":

    os.makedirs("results", exist_ok=True)

    t_end = 30
    media_interval = 2

    # --------------------------------------------------------
    # Baseline simulation
    # --------------------------------------------------------

    t, Y, media_days = simulate_with_media_change(
        p,
        t_end=t_end,
        media_interval=media_interval,
    )

    C, P, A, H = Y
    Rf, Rm, Rd, Rout = rate_arrays(t, Y, p)

    print_metrics("Baseline metrics", t, Y)

    # ========================================================
    # FIGURE 1: Baseline dynamics
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 8.2),
        sharex=True,
    )

    ax = axes[0]
    ax.plot(t, C, label="Calcium $C$")
    ax.plot(t, P, label="Phosphate $P$")
    add_media_lines(ax, media_days)
    ax.set_ylabel("Ion concentration [mM]")
    ax.legend(frameon=True)

    ax = axes[1]
    ax.plot(t, A, label="ACP precursor $A$")
    ax.plot(t, H, label="Apatite-like mineral $H$")
    add_media_lines(ax, media_days)
    ax.set_ylabel("Mineral amount\n[phosphate-equivalent]")
    ax.legend(frameon=True)

    ax = axes[2]
    ax.plot(t, Rf, label="Precursor formation $R_f$")
    ax.plot(t, Rm, label="Maturation $R_m$")
    ax.plot(t, Rd, label="Precursor dissolution $R_d$")
    ax.plot(t, Rout, label="Precursor removal $R_{out}$")
    add_media_lines(ax, media_days)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Rate\n[phosphate-equivalent/day]")
    ax.legend(frameon=True)

    save_figure(fig, "results/fig1_baseline_dynamics.pdf")

    # ========================================================
    # FIGURE 2: Vesicle and Mg effects
    # ========================================================

    Nv_values = [0.5, 1.0, 2.0, 4.0]
    Mg_values = [0.0, 0.5, 1.0, 2.0]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 8.2),
        sharex=True,
    )

    # Vesicle effect on mature mineral
    ax = axes[0]

    for Nv in Nv_values:
        p_temp = p.copy()
        p_temp["N_v"] = Nv

        t_temp, Y_temp, _ = simulate_with_media_change(
            p_temp,
            t_end=t_end,
            media_interval=media_interval,
        )

        ax.plot(t_temp, Y_temp[3], label=fr"$N_v={Nv}$")

    add_media_lines(ax, media_days)
    ax.set_ylabel("Mature mineral $H$")
    ax.legend(frameon=True)

    # Mg effect on ACP precursor
    ax = axes[1]

    for Mg in Mg_values:
        p_temp = p.copy()
        p_temp["Mg"] = Mg

        t_temp, Y_temp, _ = simulate_with_media_change(
            p_temp,
            t_end=t_end,
            media_interval=media_interval,
        )

        ax.plot(t_temp, Y_temp[2], label=fr"$Mg={Mg}$ mM")

    add_media_lines(ax, media_days)
    ax.set_ylabel("ACP precursor $A$")
    ax.legend(frameon=True)

    # Mg effect on mature mineral
    ax = axes[2]

    for Mg in Mg_values:
        p_temp = p.copy()
        p_temp["Mg"] = Mg

        t_temp, Y_temp, _ = simulate_with_media_change(
            p_temp,
            t_end=t_end,
            media_interval=media_interval,
        )

        ax.plot(t_temp, Y_temp[3], label=fr"$Mg={Mg}$ mM")

    add_media_lines(ax, media_days)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Mature mineral $H$")
    ax.legend(frameon=True)

    save_figure(fig, "results/fig2_vesicle_mg_effects.pdf")

    # ========================================================
    # FIGURE 3: Citrate effects
    # ========================================================

    Cit_values = [0.0, 0.1, 0.5, 1.0]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
    )

    # Citrate effect on ACP precursor
    ax = axes[0]

    for Cit in Cit_values:
        p_temp = p.copy()
        p_temp["Cit"] = Cit

        t_temp, Y_temp, _ = simulate_with_media_change(
            p_temp,
            t_end=t_end,
            media_interval=media_interval,
        )

        ax.plot(t_temp, Y_temp[2], label=fr"$Cit={Cit}$ mM")

    add_media_lines(ax, media_days)
    ax.set_ylabel("ACP precursor $A$")
    ax.legend(frameon=True)

    # Citrate effect on mature mineral
    ax = axes[1]

    for Cit in Cit_values:
        p_temp = p.copy()
        p_temp["Cit"] = Cit

        t_temp, Y_temp, _ = simulate_with_media_change(
            p_temp,
            t_end=t_end,
            media_interval=media_interval,
        )

        ax.plot(t_temp, Y_temp[3], label=fr"$Cit={Cit}$ mM")

    add_media_lines(ax, media_days)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Mature mineral $H$")
    ax.legend(frameon=True)

    save_figure(fig, "results/fig3_citrate_effects.pdf")

    # ========================================================
    # FIGURE 4: Combined Mg + citrate stabilization
    # ========================================================

    combined_conditions = [
        {"Mg": 0.0, "Cit": 0.0, "label": "control"},
        {"Mg": 1.0, "Cit": 0.0, "label": "Mg"},
        {"Mg": 0.0, "Cit": 0.5, "label": "citrate"},
        {"Mg": 1.0, "Cit": 0.5, "label": "Mg + citrate"},
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
    )

    ax = axes[0]

    for cond in combined_conditions:
        p_temp = p.copy()
        p_temp["Mg"] = cond["Mg"]
        p_temp["Cit"] = cond["Cit"]

        t_temp, Y_temp, _ = simulate_with_media_change(
            p_temp,
            t_end=t_end,
            media_interval=media_interval,
        )

        ax.plot(t_temp, Y_temp[2], label=cond["label"])

    add_media_lines(ax, media_days)
    ax.set_ylabel("ACP precursor $A$")
    ax.legend(frameon=True)

    ax = axes[1]

    for cond in combined_conditions:
        p_temp = p.copy()
        p_temp["Mg"] = cond["Mg"]
        p_temp["Cit"] = cond["Cit"]

        t_temp, Y_temp, _ = simulate_with_media_change(
            p_temp,
            t_end=t_end,
            media_interval=media_interval,
        )

        ax.plot(t_temp, Y_temp[3], label=cond["label"])

    add_media_lines(ax, media_days)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Mature mineral $H$")
    ax.legend(frameon=True)

    save_figure(fig, "results/fig4_combined_mg_citrate_effects.pdf")

    # ========================================================
    # Optional single-panel PDFs
    # ========================================================

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, C, label="Calcium $C$")
    ax.plot(t, P, label="Phosphate $P$")
    add_media_lines(ax, media_days)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Reactive ion concentration [mM]")
    ax.legend(frameon=True)
    save_figure(fig, "results/01_ions_media_change.pdf")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, A, label="ACP precursor $A$")
    ax.plot(t, H, label="Apatite-like mineral $H$")
    add_media_lines(ax, media_days)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Mineral amount [phosphate-equivalent]")
    ax.legend(frameon=True)
    save_figure(fig, "results/02_mineral_dynamics_media_change.pdf")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, Rf, label="Precursor formation $R_f$")
    ax.plot(t, Rm, label="Maturation $R_m$")
    ax.plot(t, Rd, label="Precursor dissolution $R_d$")
    ax.plot(t, Rout, label="Precursor removal $R_{out}$")
    add_media_lines(ax, media_days)
    ax.set_xlabel("Time [days]")
    ax.set_ylabel("Rate [phosphate-equivalent/day]")
    ax.legend(frameon=True)
    save_figure(fig, "results/02b_rates.pdf")

    print("\nSaved PDF and PNG figures to: results/")