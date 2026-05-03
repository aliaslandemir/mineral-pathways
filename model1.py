import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# ============================================================
# 1. Parameters
# ============================================================

p = {
    # External ion levels [mM]
    "C_ext": 1.2,
    "P_ext": 1.0,

    # Ion supply rates [1/day]
    "k_C": 2.0,
    "k_P": 2.0,

    # Effective Ca/P ratios
    "r_A": 1.35,
    "r_H": 1.65,

    # Supersaturation parameters
    "alpha": 3.0,
    "beta": 2.0,
    "K_A": 0.8,

    # Precursor formation
    "N_v": 1.0,
    "k_f": 0.15,
    "Phi_f": 1.0,
    "n": 2.0,

    # Maturation
    "k_m": 0.35,
    "Phi_m": 1.0,

    # Calcium limitation for maturation
    "K_Cm": 0.2,

    # Precursor loss
    "k_loss": 0.03,
}


# ============================================================
# 2. Rate laws
# ============================================================

def supersaturation(C, P, p):
    return (C ** p["alpha"] * P ** p["beta"]) / p["K_A"]


def precursor_formation(C, P, p):
    S_A = supersaturation(C, P, p)

    if S_A <= 1:
        return 0.0

    return (
        p["N_v"]
        * p["k_f"]
        * p["Phi_f"]
        * (S_A - 1) ** p["n"]
    )


def maturation(C, A, p):
    calcium_factor = C / (C + p["K_Cm"])

    return (
        p["k_m"]
        * p["Phi_m"]
        * A
        * calcium_factor
    )


def precursor_loss(A, p):
    return p["k_loss"] * A


# ============================================================
# 3. ODE system
# ============================================================

def mineralization_model(t, y, p):
    C, P, A, H = y

    # prevent numerical negative values
    C = max(C, 0.0)
    P = max(P, 0.0)
    A = max(A, 0.0)

    R_f = precursor_formation(C, P, p)
    R_m = maturation(C, A, p)
    R_l = precursor_loss(A, p)

    dCdt = (
        p["k_C"] * (p["C_ext"] - C)
        - p["r_A"] * R_f
        - (p["r_H"] - p["r_A"]) * R_m
    )

    dPdt = (
        p["k_P"] * (p["P_ext"] - P)
        - R_f
    )

    dAdt = R_f - R_m - R_l
    dHdt = R_m

    return [dCdt, dPdt, dAdt, dHdt]


# ============================================================
# 4. Simulation
# ============================================================

t_span = (0, 14)                 # days
t_eval = np.linspace(0, 14, 400)

y0 = [
    p["C_ext"],   # C initial
    p["P_ext"],   # P initial
    0.0,          # ACP precursor
    0.0,          # mature mineral
]

sol = solve_ivp(
    mineralization_model,
    t_span,
    y0,
    args=(p,),
    t_eval=t_eval,
    method="RK45",
)

t = sol.t
C, P, A, H = sol.y


# ============================================================
# 5. Simple figures
# ============================================================

plt.figure(figsize=(7, 4))
plt.plot(t, C, label="Calcium C")
plt.plot(t, P, label="Phosphate P")
plt.xlabel("Time [days]")
plt.ylabel("Reactive ion concentration [mM]")
plt.legend()
plt.tight_layout()
plt.show()


plt.figure(figsize=(7, 4))
plt.plot(t, A, label="ACP precursor A")
plt.plot(t, H, label="Apatite-like mineral H")
plt.xlabel("Time [days]")
plt.ylabel("Mineral amount [phosphate-equivalent]")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 6. Simple parameter comparison: vesicle activity
# ============================================================

Nv_values = [0.5, 1.0, 2.0, 4.0]

plt.figure(figsize=(7, 4))

for Nv in Nv_values:
    p_temp = p.copy()
    p_temp["N_v"] = Nv

    sol_temp = solve_ivp(
        mineralization_model,
        t_span,
        y0,
        args=(p_temp,),
        t_eval=t_eval,
    )

    plt.plot(sol_temp.t, sol_temp.y[3], label=f"N_v = {Nv}")

plt.xlabel("Time [days]")
plt.ylabel("Mature mineral H")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. Simple parameter comparison: precursor stabilization
# ============================================================

Phi_m_values = [1.0, 0.5, 0.25, 0.1]

plt.figure(figsize=(7, 4))

for Phi_m in Phi_m_values:
    p_temp = p.copy()
    p_temp["Phi_m"] = Phi_m

    sol_temp = solve_ivp(
        mineralization_model,
        t_span,
        y0,
        args=(p_temp,),
        t_eval=t_eval,
    )

    plt.plot(sol_temp.t, sol_temp.y[0], alpha=0.0)  # keeps axes stable
    plt.plot(sol_temp.t, sol_temp.y[2], label=f"Phi_m = {Phi_m}")

plt.xlabel("Time [days]")
plt.ylabel("ACP precursor A")
plt.legend()
plt.tight_layout()
plt.show()