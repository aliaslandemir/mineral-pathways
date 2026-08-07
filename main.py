'''
python main.py \
    --t-end 30 \
    --dt 0.05 \
    --media-interval 2 \
    --media-renewal-fraction 1 \
    --n-runs 100 \
    --jobs 6 \
    --random-seed 42 \
    --solver-method RK45 \
    --solver-rtol 1e-6 \
    --solver-atol 1e-9 \
    --C-ext 1.8018 \
    --P-ext 0.906 \
    --beta-GP 10.0 \
    --eta-48h 0.22 \
    --alpha 3 \
    --beta 2 \
    --n 1 \
    --K-A 18.0 \
    --r-A-min 1.20 \
    --r-A-max 1.49 \
    --r-H-min 1.50 \
    --r-H-max 1.67 \
    --Xi 1.0 \
    --k-f 1.0 \
    --M-max 2.0 \
    --k-m 0.2 \
    --a-AA0 0.28 \
    --t-AA-half-life 24.0 \
    --t-AA-half-life-levels 6,12,18,24,36,48,72 \
    --beta-GP-levels 0,1,2,3,4,5,7.5,10,15,20 \
    --M-max-levels 1,2,3,4,6,8 \
    --eta-48h-levels 0.05,0.10,0.18,0.22,0.25,0.33,0.40,0.50,0.65 \
    --K-A-levels 8,10,12,15,18,20,22,24 \
    --k-f-levels 0.10,0.25,0.50,0.75,1.00,1.50,2.00 \
    --k-m-levels 0.01,0.05,0.10,0.20,0.50,1.00,2.00
'''


import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import (
    PLOT_DEFAULTS,
    RUN_DEFAULTS,
    STOCHASTIC_DEFAULTS,
    PlotDefaults,
    build_base_params,
)
from src.io_utils import create_run_directory, parse_figure_formats
from src.model import (
    synchronize_hydrolysis_parameters,
    simulate,
    transient_metrics,
)
from src.sampling import apply_stochastic_variability, build_ca_p_samples, scaled_unit_sample
from src.sensitivity import run_stochastic_parameter_sweep
from src.visualization import configure_matplotlib, generate_stochastic_figure_set


class ArgumentDefaultsRawFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass


def build_plot_config(args: argparse.Namespace) -> PlotDefaults:
    phase_lines = tuple(np.arange(args.media_interval, args.t_end, args.media_interval, dtype=float))
    return PlotDefaults(
        save_dpi=args.save_dpi,
        figure_dpi=args.figure_dpi,
        font_size=args.font_size,
        axes_title_size=args.axes_title_size,
        axes_label_size=args.axes_label_size,
        legend_font_size=args.legend_font_size,
        tick_label_size=args.tick_label_size,
        phase_lines_days=phase_lines,
        figure_formats=parse_figure_formats(args.figure_formats),
    )


def parse_float_list(text: str) -> list[float]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of numbers.")
    try:
        return [float(v) for v in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric list: {text}") from exc


def build_parser() -> argparse.ArgumentParser:
    defaults = build_base_params()
    parser = argparse.ArgumentParser(
        description="Run the ACP-to-apatite reduced-order model and generate figures.",
        formatter_class=ArgumentDefaultsRawFormatter,
        epilog=(
        "Examples:\n"
        "  python3 main.py\n"
            "  python3 main.py --a-AA0 0.28 --t-AA-half-life 24.0\n"
            "  python3 main.py --M-max 2.0 --k-m 0.2 --n-runs 50\n"
            "  python3 main.py --t-AA-half-life-levels 6,12,18,24,36,48,72\n"
        ),
    )

    run = parser.add_argument_group("run controls")
    run.add_argument("--t-end", "--t_end", dest="t_end", type=float, default=RUN_DEFAULTS.t_end)
    run.add_argument("--dt", type=float, default=RUN_DEFAULTS.dt)
    run.add_argument("--media-interval", "--media_interval", dest="media_interval", type=float, default=RUN_DEFAULTS.media_interval)
    run.add_argument(
        "--media-renewal-fraction",
        "--media_renewal_fraction",
        dest="media_renewal_fraction",
        type=float,
        default=defaults["media_renewal_fraction"],
        help="Fraction of soluble medium replaced at renewal; 0 retains all soluble carryover and 1 is full replacement.",
    )
    run.add_argument("--n-runs", "--n_runs", dest="n_runs", type=int, default=RUN_DEFAULTS.n_runs)
    run.add_argument("--jobs", type=int, default=RUN_DEFAULTS.jobs, help="Worker processes for independent parameter-sweep simulations; use 1 for serial execution.")
    run.add_argument("--seed", type=int, default=RUN_DEFAULTS.seed)
    run.add_argument(
        "--random-seed",
        "--random_seed",
        dest="random_seed",
        nargs="?",
        const="fresh",
        default="fresh",
        help=(
            "Set the stochastic seed when a value is supplied, or generate a fresh "
            "seed when supplied without a value."
        ),
    )
    run.add_argument("--solver-method", "--solver_method", dest="solver_method", type=str, default=RUN_DEFAULTS.solver_method)
    run.add_argument("--solver-rtol", "--solver_rtol", dest="solver_rtol", type=float, default=RUN_DEFAULTS.solver_rtol)
    run.add_argument("--solver-atol", "--solver_atol", dest="solver_atol", type=float, default=RUN_DEFAULTS.solver_atol)

    ions = parser.add_argument_group("medium and ion parameters")
    ions.add_argument("--C-ext", "--C_ext", dest="C_ext", type=float, default=defaults["C_ext"])
    ions.add_argument("--P-ext", "--P_ext", dest="P_ext", type=float, default=defaults["P_ext"])
    ions.add_argument("--beta-GP", "--beta_GP", dest="beta_GP", type=float, default=defaults["beta_GP"])
    ions.add_argument(
        "--eta-48h",
        "--eta_48h",
        dest="eta_48h",
        type=float,
        default=defaults["eta_48h"],
        help="Reference beta-GP fraction hydrolyzed after 48 h; converted exactly to eta_dt for the selected --dt.",
    )
    hydrolysis_override = ions.add_mutually_exclusive_group()
    hydrolysis_override.add_argument(
        "--eta-dt",
        "--eta_dt",
        dest="eta_dt",
        type=float,
        default=None,
        help="Optional fraction of remaining beta-GP hydrolyzed during one --dt interval; converted exactly to k_hyd.",
    )
    hydrolysis_override.add_argument("--k-hyd", "--k_hyd", dest="k_hyd", type=float, default=None, help="Optional beta-GP hydrolysis first-order rate constant (day^-1).")

    formation = parser.add_argument_group("ACP formation parameters")
    formation.add_argument("--alpha", type=float, default=defaults["alpha"])
    formation.add_argument("--beta", type=float, default=defaults["beta"])
    formation.add_argument("--n", type=float, default=defaults["n"])
    formation.add_argument("--K-A", "--K_A", dest="K_A", type=float, default=defaults["K_A"])
    formation.add_argument("--k-f", "--k_f", dest="k_f", type=float, default=defaults["k_f"])
    formation.add_argument("--M-max", "--M_max", dest="M_max", type=float, default=defaults["M_max"], help="Maximum retained matrix-associated mineral capacity in the same arbitrary model units as A and H.")

    maturation = parser.add_argument_group("ACP maturation parameters")
    maturation.add_argument("--k-m", "--k_m", dest="k_m", type=float, default=defaults["k_m"])
    maturation.add_argument("--a-AA0", "--a_AA0", dest="a_AA0", type=float, default=defaults["a_AA0"], help="Initial AA2P/ascorbate concentration after each medium renewal (mM).")
    maturation.add_argument("--t-AA-half-life", "--t_AA_half_life", dest="t_AA_half_life", type=float, default=defaults["t_AA_half_life"], help="AA2P/ascorbate decay half-life after each medium renewal (hours).")

    stoich = parser.add_argument_group("stoichiometry stochastic ranges")
    stoich.add_argument("--r-A-min", "--r_A_min", dest="r_A_min", type=float, default=STOCHASTIC_DEFAULTS.r_A_min)
    stoich.add_argument("--r-A-max", "--r_A_max", dest="r_A_max", type=float, default=STOCHASTIC_DEFAULTS.r_A_max)
    stoich.add_argument("--r-H-min", "--r_H_min", dest="r_H_min", type=float, default=STOCHASTIC_DEFAULTS.r_H_min)
    stoich.add_argument("--r-H-max", "--r_H_max", dest="r_H_max", type=float, default=STOCHASTIC_DEFAULTS.r_H_max)
    stoich.add_argument("--ca-p-variability-scale", "--Xi", "--xi", dest="ca_p_variability_scale", type=float, default=STOCHASTIC_DEFAULTS.Xi)

    ranges = parser.add_argument_group("sweep ranges")
    ranges.add_argument("--t-AA-half-life-levels", "--t_AA_half_life_levels", dest="t_AA_half_life_levels", type=parse_float_list, default=list(STOCHASTIC_DEFAULTS.t_AA_half_life_levels))
    ranges.add_argument("--beta-GP-levels", "--beta_GP_levels", dest="beta_GP_levels", type=parse_float_list, default=list(STOCHASTIC_DEFAULTS.beta_gp_levels))
    ranges.add_argument("--eta-48h-levels", "--eta_48h_levels", dest="eta_48h_levels", type=parse_float_list, default=list(STOCHASTIC_DEFAULTS.eta_48h_levels))
    ranges.add_argument("--K-A-levels", "--K_A_levels", dest="K_A_levels", type=parse_float_list, default=list(STOCHASTIC_DEFAULTS.K_A_levels))
    ranges.add_argument("--k-f-levels", "--k_f_levels", dest="k_f_levels", type=parse_float_list, default=list(STOCHASTIC_DEFAULTS.k_f_levels))
    ranges.add_argument("--M-max-levels", "--M_max_levels", dest="M_max_levels", type=parse_float_list, default=list(STOCHASTIC_DEFAULTS.M_max_levels))
    ranges.add_argument("--k-m-levels", "--k_m_levels", dest="k_m_levels", type=parse_float_list, default=list(STOCHASTIC_DEFAULTS.k_m_levels))

    plot = parser.add_argument_group("plot controls")
    plot.add_argument("--save-dpi", "--save_dpi", dest="save_dpi", type=int, default=PLOT_DEFAULTS.save_dpi)
    plot.add_argument("--figure-dpi", "--figure_dpi", dest="figure_dpi", type=int, default=PLOT_DEFAULTS.figure_dpi)
    plot.add_argument("--font-size", "--font_size", dest="font_size", type=float, default=PLOT_DEFAULTS.font_size)
    plot.add_argument("--axes-title-size", "--axes_title_size", dest="axes_title_size", type=float, default=PLOT_DEFAULTS.axes_title_size)
    plot.add_argument("--axes-label-size", "--axes_label_size", dest="axes_label_size", type=float, default=PLOT_DEFAULTS.axes_label_size)
    plot.add_argument("--legend-font-size", "--legend_font_size", dest="legend_font_size", type=float, default=PLOT_DEFAULTS.legend_font_size)
    plot.add_argument("--tick-label-size", "--tick_label_size", dest="tick_label_size", type=float, default=PLOT_DEFAULTS.tick_label_size)
    plot.add_argument(
        "--figure-formats",
        default=",".join(PLOT_DEFAULTS.figure_formats),
        help="Comma-separated figure formats to save: pdf,png. Use an empty value to skip figure files.",
    )
    return parser


def _scaled_unit_sample(u: float, stochasticity: float) -> float:
    return scaled_unit_sample(u, stochasticity)


def map_cli_xi_to_ra_rh(u_a: float, u_h: float, args: argparse.Namespace) -> tuple[float, float, float, float]:
    xi_a = _scaled_unit_sample(u_a, args.ca_p_variability_scale)
    xi_h = _scaled_unit_sample(u_h, args.ca_p_variability_scale)
    r_a = args.r_A_min + (args.r_A_max - args.r_A_min) * xi_a
    r_h = args.r_H_min + (args.r_H_max - args.r_H_min) * xi_h
    return float(r_a), float(r_h), float(xi_a), float(xi_h)


def build_stochastic_samples(args: argparse.Namespace):
    return build_ca_p_samples(
        n_runs=args.n_runs,
        seed=args.seed,
        r_A_min=args.r_A_min,
        r_A_max=args.r_A_max,
        r_H_min=args.r_H_min,
        r_H_max=args.r_H_max,
        xi_scale=args.ca_p_variability_scale,
    )


def resolve_run_seed(args: argparse.Namespace) -> None:
    if args.random_seed is None:
        return
    if args.random_seed == "fresh":
        args.seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        return
    args.seed = int(args.random_seed)


def apply_cli_parameters(params, args: argparse.Namespace):
    p = dict(params)
    for name in [
        "C_ext",
        "P_ext",
        "beta_GP",
        "eta_48h",
        "alpha",
        "beta",
        "n",
        "K_A",
        "k_f",
        "M_max",
        "k_m",
        "a_AA0",
        "t_AA_half_life",
        "solver_rtol",
        "solver_atol",
        "media_renewal_fraction",
    ]:
        p[name] = float(getattr(args, name))
    p.pop("_k_hyd_explicit", None)
    p.pop("_eta_dt_explicit", None)
    p.pop("_beta_GP_cycle_explicit", None)
    if args.k_hyd is not None:
        p["k_hyd"] = float(args.k_hyd)
        p["_k_hyd_explicit"] = True
    elif args.eta_dt is not None:
        p["eta_dt"] = float(args.eta_dt)
        p["eta_dt_interval_days"] = float(args.dt)
        p["_eta_dt_explicit"] = True
    else:
        p.pop("k_hyd", None)
    synchronize_hydrolysis_parameters(p, args.dt, args.media_interval)
    p["solver_method"] = str(args.solver_method)
    return p


def fmt_result(value: float) -> str:
    return f"{float(value):.6g}"


def ensure_level_in_sweep(levels: list[float], baseline_value: float, rtol: float = 1e-9, atol: float = 1e-12) -> list[float]:
    if any(np.isclose(float(level), float(baseline_value), rtol=rtol, atol=atol) for level in levels):
        return levels
    return sorted([*levels, float(baseline_value)])


def log2_fold_change(final_values: np.ndarray, baseline_final_values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    final_values = np.asarray(final_values, dtype=float)
    baseline_final_values = np.asarray(baseline_final_values, dtype=float)
    if final_values.shape == baseline_final_values.shape:
        denom = baseline_final_values
    else:
        denom = np.full_like(final_values, float(np.nanmean(baseline_final_values)))
    return np.log2((final_values + eps) / (denom + eps))


def write_sweep_log2fc_summary(
    out_dir: Path,
    baseline_h_final_runs: np.ndarray,
    sweep_specs: list[tuple[str, str, list[float], dict[float, np.ndarray]]],
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    baseline_mean = float(np.nanmean(baseline_h_final_runs))
    for parameter, units, levels, runs_by_level in sweep_specs:
        for level in levels:
            h_final = np.asarray(runs_by_level[level], dtype=float)[:, -1]
            log2fc = log2_fold_change(h_final, baseline_h_final_runs)
            rows.append(
                {
                    "parameter": parameter,
                    "level": float(level),
                    "units": units,
                    "baseline_H_final_mean": baseline_mean,
                    "H_final_mean": float(np.nanmean(h_final)),
                    "H_final_sd": float(np.nanstd(h_final)),
                    "log2FC_H_final_mean": float(np.nanmean(log2fc)),
                    "log2FC_H_final_sd": float(np.nanstd(log2fc)),
                    "n_runs": int(h_final.size),
                }
            )

    for filename in ("sweep_log2fc_summary.csv", "sweep_results.csv"):
        with (out_dir / filename).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def write_summary(
    out_dir: Path,
    params,
    metrics,
    stochastic_samples,
    t_AA_half_life_sweep,
    beta_gp_sweep,
    eta_48h_sweep,
    K_A_sweep,
    k_f_sweep,
    M_max_sweep,
    k_m_sweep,
    args: argparse.Namespace,
    sweep_log2fc_rows: list[dict[str, float | str]] | None = None,
):
    lines = []
    lines.append("# ACP-Apatite Model Run\n")
    lines.append(f"Run UTC: {datetime.now(timezone.utc).isoformat()}\n")
    lines.append(f"Command: `{' '.join(sys.argv)}`\n")
    lines.append("## Core transient metrics\n")
    for k, v in metrics.items():
        if isinstance(v, str):
            lines.append(f"- {k}: {v}\n")
        else:
            lines.append(f"- {k}: {v:.6g}\n")

    lines.append("\n## Baseline parameters\n")
    for k, v in params.items():
        lines.append(f"- {k}: {v}\n")

    lines.append("\n## Hydrolysis timestep mapping\n")
    lines.append(f"- eta_48h_reference: {float(params['eta_48h']):.8g}\n")
    lines.append(f"- eta_dt: {float(params['eta_dt']):.8g}\n")
    lines.append(f"- eta_dt_interval_days: {float(params['eta_dt_interval_days']):.8g}\n")
    lines.append(f"- k_hyd_day^-1: {float(params['k_hyd']):.8g}\n")
    lines.append("- relation: eta_dt = 1 - (1 - eta_48h)^(dt / 2)\n")
    lines.append(f"- beta_GP_external_mM: {float(params['beta_GP']):.8g}\n")
    lines.append(f"- beta_GP_cycle_release_mM: {float(params['beta_GP_cycle_release']):.8g}\n")
    lines.append("- relation: q_Pi = beta_GP * (1 - exp(-k_hyd * media_interval)); the reservoir is not rescaled\n")

    lines.append("\n## Executable CLI controls\n")
    for k, v in sorted(vars(args).items()):
        lines.append(f"- {k}: {v}\n")

    if stochastic_samples:
        lines.append("\n## Stochastic samples\n")
        lines.append(f"- n_runs: {len(stochastic_samples)}\n")
        lines.append(f"- r_A_range: [{args.r_A_min}, {args.r_A_max}]\n")
        lines.append(f"- r_H_range: [{args.r_H_min}, {args.r_H_max}]\n")
        lines.append(f"- ca_p_variability_scale: {args.ca_p_variability_scale}\n")
        lines.append("- effective xi_A, xi_H = 0.5 + Xi * (U - 0.5), U in [0, 1]\n")
        lines.append("- stochasticity is limited to mineral stoichiometry; raw ACP/H kinetic bands are not broad kinetic uncertainty bands\n")
        lines.append('- uncertainty_source: "sampled effective Ca/P ratios only"\n')
        for name in ["r_A", "r_H", "xi_A", "xi_H"]:
            vals = np.array([s[name] for s in stochastic_samples], dtype=float)
            lines.append(f"- {name}_mean_std: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n")

    lines.append("\n## AA2P/ascorbate half-life sensitivity: H_final\n")
    for v, sim, _ in t_AA_half_life_sweep:
        lines.append(f"- t_AA_half_life={v:.6g} h -> Rm_peak={fmt_result(np.max(sim.Rm))}, H_final={fmt_result(sim.H[-1])}\n")

    lines.append("\n## beta-GP sensitivity: final reactive P and H_final\n")
    for v, sim, _ in beta_gp_sweep:
        lines.append(f"- beta_GP={v:.6g} mM -> P_final={fmt_result(sim.P[-1])}, H_final={fmt_result(sim.H[-1])}\n")

    lines.append("\n## eta_48h sensitivity: final reactive P and H_final\n")
    for v, sim, _ in eta_48h_sweep:
        lines.append(f"- eta_48h={v:.6g} -> P_final={fmt_result(sim.P[-1])}, B_final={fmt_result(sim.B[-1])}, H_final={fmt_result(sim.H[-1])}\n")

    lines.append("\n## K_A sensitivity: onset and H_final\n")
    for v, sim, metric in K_A_sweep:
        lines.append(f"- K_A={v:.6g} -> onset_SA={fmt_result(metric['onset_time_SA_days'])}, H_final={fmt_result(sim.H[-1])}\n")

    lines.append("\n## k_f sensitivity: peak formation rate and H_final\n")
    for v, sim, _ in k_f_sweep:
        lines.append(f"- k_f={v:.6g} -> Rf_peak={fmt_result(np.max(sim.Rf))}, H_final={fmt_result(sim.H[-1])}\n")

    lines.append("\n## M_max sensitivity: retained capacity and H_final\n")
    for v, sim, _ in M_max_sweep:
        lines.append(f"- M_max={v:.6g} -> A_plus_H_final={fmt_result(sim.A[-1] + sim.H[-1])}, H_final={fmt_result(sim.H[-1])}\n")

    lines.append("\n## k_m sensitivity: peak maturation rate and H_final\n")
    for v, sim, _ in k_m_sweep:
        lines.append(f"- k_m={v:.6g} -> Rm_peak={fmt_result(np.max(sim.Rm))}, H_final={fmt_result(sim.H[-1])}\n")

    if sweep_log2fc_rows:
        lines.append("\n## Sweep endpoint results: log2 fold change in day-28 H vs baseline\n")
        for row in sweep_log2fc_rows:
            lines.append(
                f"- {row['parameter']}={float(row['level']):.6g} {row['units']} -> "
                f"H_final_mean={float(row['H_final_mean']):.6g}, "
                f"log2FC={float(row['log2FC_H_final_mean']):.4g} ± {float(row['log2FC_H_final_sd']):.4g}\n"
            )

    (out_dir / "summary.md").write_text("".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    if args.n_runs < 1:
        raise ValueError("--n-runs must be at least 1.")
    if args.jobs < 1:
        raise ValueError("--jobs must be at least 1.")
    if args.M_max <= 0.0:
        raise ValueError("--M-max must be positive.")
    if not 0.0 <= args.eta_48h < 1.0:
        raise ValueError("--eta-48h must be in [0, 1).")
    if args.eta_dt is not None and not 0.0 <= args.eta_dt < 1.0:
        raise ValueError("--eta-dt must be in [0, 1).")
    if args.k_hyd is not None and args.k_hyd < 0.0:
        raise ValueError("--k-hyd must be non-negative.")
    resolve_run_seed(args)
    plot_config = build_plot_config(args)
    configure_matplotlib(plot_config)

    p = apply_cli_parameters(build_base_params(), args)
    out_dir = create_run_directory("main", "run")

    stochastic_samples = build_stochastic_samples(args)
    baseline = apply_stochastic_variability(p, stochastic_samples[0])
    runs = []
    metrics_list = []
    for sample in stochastic_samples:
        pr = apply_stochastic_variability(p, sample)
        sim = simulate(pr, t_end=args.t_end, dt=args.dt, media_interval=args.media_interval)
        runs.append(sim)
        metrics_list.append(transient_metrics(sim))

    t_days = runs[0].t
    ions_C = np.array([r.C for r in runs])
    ions_P = np.array([r.P for r in runs])
    phosphate_input_params = dict(p)
    phosphate_input_params["k_f"] = 0.0
    phosphate_input_reference = simulate(
        phosphate_input_params,
        t_end=args.t_end,
        dt=args.dt,
        media_interval=args.media_interval,
    )
    ions_P_reference = np.array([phosphate_input_reference.P])
    mineral_A = np.array([r.A for r in runs])
    mineral_H = np.array([r.H for r in runs])
    rates_Rf = np.array([r.Rf for r in runs])
    rates_Rm = np.array([r.Rm for r in runs])
    saturation_A = np.array([r.S_A for r in runs])
    metrics = {
        "A_peak_mean": float(np.mean([m["A_peak"] for m in metrics_list])),
        "A_peak_std": float(np.std([m["A_peak"] for m in metrics_list])),
        "t_peak_days_mean": float(np.mean([m["t_peak_days"] for m in metrics_list])),
        "t_peak_days_std": float(np.std([m["t_peak_days"] for m in metrics_list])),
        "H_final_mean": float(np.mean([m["H_final"] for m in metrics_list])),
        "H_final_std": float(np.std([m["H_final"] for m in metrics_list])),
        "A_final_mean": float(np.mean([m["A_final"] for m in metrics_list])),
        "A_final_std": float(np.std([m["A_final"] for m in metrics_list])),
        "decline_fraction_mean": float(np.mean([m["decline_fraction"] for m in metrics_list])),
        "onset_time_SA_days_mean": float(np.nanmean([m["onset_time_SA_days"] for m in metrics_list])),
        "AUC_A_mean": float(np.mean([m["AUC_A"] for m in metrics_list])),
        "integrated_maturation_fraction_mean": float(np.mean([m["integrated_maturation_fraction"] for m in metrics_list])),
        "phi_H_final_mean": float(np.mean([m["phi_H_final"] for m in metrics_list])),
        "pulse_index_mean": float(np.mean([m["pulse_index"] for m in metrics_list])),
    }
    t_AA_half_life_levels = ensure_level_in_sweep(list(args.t_AA_half_life_levels), p["t_AA_half_life"])
    t_AA_half_life_result = run_stochastic_parameter_sweep(
        p, "t_AA_half_life", t_AA_half_life_levels, stochastic_samples, ("H", "Rm"), args.t_end, args.dt,
        args.media_interval, baseline_runs=runs, jobs=args.jobs,
    )
    t_AA_half_life_sweep = t_AA_half_life_result.representative
    t_AA_half_life_h_runs = t_AA_half_life_result.trajectories["H"]
    t_AA_half_life_rm_runs = t_AA_half_life_result.trajectories["Rm"]

    beta_gp_levels = ensure_level_in_sweep(list(args.beta_GP_levels), p["beta_GP"])
    beta_gp_result = run_stochastic_parameter_sweep(
        p, "beta_GP", beta_gp_levels, stochastic_samples, ("H",),
        args.t_end, args.dt, args.media_interval, baseline_runs=runs, jobs=args.jobs,
    )
    beta_gp_sweep = beta_gp_result.representative
    beta_gp_h_runs = beta_gp_result.trajectories["H"]

    eta_48h_levels = ensure_level_in_sweep(list(args.eta_48h_levels), p["eta_48h"])
    eta_48h_result = run_stochastic_parameter_sweep(
        p, "eta_48h", eta_48h_levels, stochastic_samples, ("H",),
        args.t_end, args.dt, args.media_interval, baseline_runs=runs, jobs=args.jobs,
    )
    eta_48h_sweep = eta_48h_result.representative
    eta_48h_h_runs = eta_48h_result.trajectories["H"]

    K_A_levels = ensure_level_in_sweep(list(args.K_A_levels), p["K_A"])
    K_A_result = run_stochastic_parameter_sweep(
        p, "K_A", K_A_levels, stochastic_samples, ("H",),
        args.t_end, args.dt, args.media_interval, baseline_runs=runs, jobs=args.jobs,
    )
    K_A_sweep = K_A_result.representative
    K_A_h_runs = K_A_result.trajectories["H"]

    k_f_levels = ensure_level_in_sweep(list(args.k_f_levels), p["k_f"])
    k_f_result = run_stochastic_parameter_sweep(
        p, "k_f", k_f_levels, stochastic_samples, ("H",),
        args.t_end, args.dt, args.media_interval, baseline_runs=runs, jobs=args.jobs,
    )
    k_f_sweep = k_f_result.representative
    k_f_h_runs = k_f_result.trajectories["H"]

    M_max_levels = ensure_level_in_sweep(list(args.M_max_levels), p["M_max"])
    M_max_result = run_stochastic_parameter_sweep(
        p, "M_max", M_max_levels, stochastic_samples, ("H",),
        args.t_end, args.dt, args.media_interval, baseline_runs=runs, jobs=args.jobs,
    )
    M_max_sweep = M_max_result.representative
    M_max_h_runs = M_max_result.trajectories["H"]

    k_m_levels = ensure_level_in_sweep(list(args.k_m_levels), p["k_m"])
    k_m_result = run_stochastic_parameter_sweep(
        p, "k_m", k_m_levels, stochastic_samples, ("H",),
        args.t_end, args.dt, args.media_interval, baseline_runs=runs, jobs=args.jobs,
    )
    k_m_sweep = k_m_result.representative
    k_m_h_runs = k_m_result.trajectories["H"]

    baseline_h_final_runs = mineral_H[:, -1]
    sweep_log2fc_rows = write_sweep_log2fc_summary(
        out_dir,
        baseline_h_final_runs,
        [
            ("t_AA_half_life", "h", t_AA_half_life_levels, t_AA_half_life_h_runs),
            ("beta_GP", "mM", beta_gp_levels, beta_gp_h_runs),
            ("eta_48h", "", eta_48h_levels, eta_48h_h_runs),
            ("K_A", "", K_A_levels, K_A_h_runs),
            ("k_f", "", k_f_levels, k_f_h_runs),
            ("M_max", "", M_max_levels, M_max_h_runs),
            ("k_m", "day^-1", k_m_levels, k_m_h_runs),
        ],
    )

    generate_stochastic_figure_set(
        out_dir=out_dir,
        t_days=t_days,
        ions_C=ions_C,
        ions_P=ions_P,
        mineral_A=mineral_A,
        mineral_H=mineral_H,
        rates_Rf=rates_Rf,
        rates_Rm=rates_Rm,
        t_AA_half_life_levels=t_AA_half_life_levels,
        t_AA_half_life_h_runs=t_AA_half_life_h_runs,
        t_AA_half_life_rm_runs=t_AA_half_life_rm_runs,
        beta_gp_levels=beta_gp_levels,
        beta_gp_h_runs=beta_gp_h_runs,
        eta_48h_levels=eta_48h_levels,
        eta_48h_h_runs=eta_48h_h_runs,
        K_A_levels=K_A_levels,
        K_A_h_runs=K_A_h_runs,
        k_f_levels=k_f_levels,
        k_f_h_runs=k_f_h_runs,
        M_max_levels=M_max_levels,
        M_max_h_runs=M_max_h_runs,
        k_m_levels=k_m_levels,
        k_m_h_runs=k_m_h_runs,
        baseline_h_final_runs=baseline_h_final_runs,
        baseline_beta_gp=p["beta_GP"],
        baseline_eta_48h=p["eta_48h"],
        baseline_K_A=p["K_A"],
        baseline_k_f=p["k_f"],
        baseline_M_max=p["M_max"],
        baseline_k_m=p["k_m"],
        plot_config=plot_config,
        saturation_A=saturation_A,
        ions_P_reference=ions_P_reference,
    )

    write_summary(
        out_dir,
        baseline,
        metrics,
        stochastic_samples,
        t_AA_half_life_sweep,
        beta_gp_sweep,
        eta_48h_sweep,
        K_A_sweep,
        k_f_sweep,
        M_max_sweep,
        k_m_sweep,
        args,
        sweep_log2fc_rows=sweep_log2fc_rows,
    )
    print(f"Saved figures and summary to: {out_dir}")


if __name__ == "__main__":
    main()
