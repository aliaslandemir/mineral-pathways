

r"""Sobol global sensitivity analysis for the ACP-to-apatite model.

Run from the repository root:

python -m src.sobol_sensitivity \
    --sobol-n-base 1024 \
    --jobs 6 \
    --save-figures \
    --figure-formats pdf,png

This uses the same baseline as the default main run:
t_end=30 days, dt=0.05 days, medium renewal every 2 days, beta_GP=10 mM,
eta_48h=0.22, K_A=18, k_f=1.0, M_max=2.0, k_m=0.2 day^-1,
a_AA0=0.28 mM, n=1, and t_AA_half_life=24 h.

Sobol ranges are defined in src.config.SensitivityDefaults and are aligned
with the main sweeps, including k_m in [0.01, 2.00] day^-1.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
import sys
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np
from SALib.analyze import sobol
from SALib.sample import sobol as sobol_sampler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PLOT_DEFAULTS, RUN_DEFAULTS, SENSITIVITY_DEFAULTS, STOCHASTIC_DEFAULTS, build_base_params
from src.io_utils import create_run_directory, parse_figure_formats, save_figure
from src.model import SimulationResult, regime_metrics, simulate, synchronize_hydrolysis_parameters
from src.visualization import clean_axis, configure_matplotlib

FIGURE_FORMATS: tuple[str, ...] = ("pdf", "png")
SOBOL_FIGURE_TOP_N = 5
RNG_SEED = RUN_DEFAULTS.seed

# Sobol ranges for the current baseline run. These are the main parameters
# requested for investigation; figures report the top 5 by total-order index.
SOBOL_BOUNDS: Dict[str, Tuple[float, float]] = dict(SENSITIVITY_DEFAULTS.sobol_bounds)
PARAM_DISPLAY = {
    "media_interval": "$T_{\\mathrm{media}}$ interval",
    "beta_GP": "$\\beta$-GP concentration",
    "eta_48h": "$\\eta_{48h}$ hydrolysis",
    "K_A": "$K_A$ threshold",
    "k_f": "$k_f$ formation",
    "M_max": "$M_{\\max}$ capacity",
    "k_m": "$k_m$ maturation",
    "t_AA_half_life": "$t_{1/2,\\mathrm{Asc}}$ half-life",
}

OUTPUT_DISPLAY = {
    "H_final": ("hfinal", "Sobol sensitivity of final apatite-like mineral accumulation"),
    "A_peak": ("acppeak", "Peak ACP accumulation"),
    "t_onset": ("onset", "Supersaturation onset time"),
    "t_A_peak": ("apeaktime", "Time of ACP peak"),
    "ACP_persistence": ("acppersistence", "ACP persistence"),
    "pulse_index": ("pulseindex", "Formation pulse index"),
    "maturation_ratio": ("maturationratio", "Maturation ratio"),
}


@dataclass
class ScalarOutputs:
    h_final: float
    a_max: float
    t_onset: float
    t_a_peak: float
    acp_persistence: float
    pulse_index: float
    maturation_ratio: float


def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    save_figure(fig, out_dir, name, FIGURE_FORMATS, dpi=600, transparent=False)


def _compute_outputs(sim: SimulationResult) -> ScalarOutputs:
    h_final = float(sim.H[-1])
    a_max = float(np.max(sim.A))
    t_a_peak = float(sim.t[int(np.argmax(sim.A))])
    metrics = regime_metrics(sim)
    t_onset = float(metrics["onset_time_SA_days"])
    if not np.isfinite(t_onset):
        t_onset = float(RUN_DEFAULTS.t_end)

    return ScalarOutputs(
        h_final=h_final,
        a_max=a_max,
        t_onset=t_onset,
        t_a_peak=t_a_peak,
        acp_persistence=float(metrics["AUC_A"]),
        pulse_index=float(metrics["pulse_index"]),
        maturation_ratio=float(metrics["maturation_ratio"]),
    )


def _simulate_with_params(base: Dict[str, float], updates: Dict[str, float]) -> Tuple[SimulationResult, ScalarOutputs]:
    p = dict(base)
    p.update(updates)
    media_interval = float(p.pop("media_interval", RUN_DEFAULTS.media_interval))
    if "eta_48h" in updates:
        p.pop("_k_hyd_explicit", None)
        p.pop("_eta_dt_explicit", None)
        synchronize_hydrolysis_parameters(p, RUN_DEFAULTS.dt, media_interval)
    sim = simulate(p, t_end=RUN_DEFAULTS.t_end, dt=RUN_DEFAULTS.dt, media_interval=media_interval)
    return sim, _compute_outputs(sim)


def _evaluate_sample_replicates(
    base_params: Dict[str, float],
    updates: Dict[str, float],
    sample_idx: int,
    replicates_per_sample: int,
) -> Tuple[float, float, float, float, float, float, float]:
    del sample_idx
    if int(replicates_per_sample) != 1:
        raise ValueError("Sobol sample responses are deterministic; use --replicates-per-sample 1.")
    _, out = _simulate_with_params(base_params, updates)
    return (
        out.h_final,
        out.a_max,
        out.t_onset,
        out.t_a_peak,
        out.acp_persistence,
        out.pulse_index,
        out.maturation_ratio,
    )


def _sobol_problem() -> Dict[str, object]:
    names = np.array(list(SOBOL_BOUNDS.keys()), dtype=object)
    return {
        "num_vars": int(len(names)),
        "names": names,
        "bounds": [list(SOBOL_BOUNDS[n]) for n in names],
    }


def _validate_sobol_n_base(n_base: int) -> int:
    n = int(n_base)
    if n <= 0 or n & (n - 1):
        raise ValueError(f"sobol_n_base must be a positive power of two; got {n_base!r}.")
    return n


def _run_sobol(base_params: Dict[str, float], n_base: int, replicates_per_sample: int = 1) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    n_base = _validate_sobol_n_base(n_base)
    problem = _sobol_problem()
    x = sobol_sampler.sample(problem, n_base, calc_second_order=False, scramble=True, seed=RNG_SEED)
    y_h = np.zeros(x.shape[0], dtype=float)
    y_a = np.zeros(x.shape[0], dtype=float)
    y_onset = np.zeros(x.shape[0], dtype=float)
    y_t_peak = np.zeros(x.shape[0], dtype=float)
    y_persistence = np.zeros(x.shape[0], dtype=float)
    y_pulse = np.zeros(x.shape[0], dtype=float)
    y_ratio = np.zeros(x.shape[0], dtype=float)

    for i in range(x.shape[0]):
        updates = {name: float(x[i, j]) for j, name in enumerate(problem["names"])}
        h, a, onset, t_peak, persistence, pulse, ratio = _evaluate_sample_replicates(
            base_params=base_params,
            updates=updates,
            sample_idx=i,
            replicates_per_sample=replicates_per_sample,
        )
        y_h[i] = h
        y_a[i] = a
        y_onset[i] = onset
        y_t_peak[i] = t_peak
        y_persistence[i] = persistence
        y_pulse[i] = pulse
        y_ratio[i] = ratio

    return problem, {
        "H_final": y_h,
        "A_peak": y_a,
        "t_onset": y_onset,
        "t_A_peak": y_t_peak,
        "ACP_persistence": y_persistence,
        "pulse_index": y_pulse,
        "maturation_ratio": y_ratio,
    }


def _evaluate_one_sample(args: Tuple[Dict[str, float], np.ndarray, List[str], int, int]) -> Tuple[float, float, float, float, float, float, float]:
    base_params, row, names, sample_idx, replicates_per_sample = args
    updates = {name: float(row[j]) for j, name in enumerate(names)}
    return _evaluate_sample_replicates(
        base_params=base_params,
        updates=updates,
        sample_idx=sample_idx,
        replicates_per_sample=replicates_per_sample,
    )


def _run_sobol_parallel(
    base_params: Dict[str, float],
    n_base: int,
    jobs: int,
    replicates_per_sample: int = 1,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    n_base = _validate_sobol_n_base(n_base)
    problem = _sobol_problem()
    x = sobol_sampler.sample(problem, n_base, calc_second_order=False, scramble=True, seed=RNG_SEED)
    names = list(problem["names"])
    with Pool(processes=jobs) as pool:
        rows = pool.map(
            _evaluate_one_sample,
            [(base_params, x[i], names, i, replicates_per_sample) for i in range(x.shape[0])],
        )
    arr = np.array(rows, dtype=float)
    return problem, {
        "H_final": arr[:, 0],
        "A_peak": arr[:, 1],
        "t_onset": arr[:, 2],
        "t_A_peak": arr[:, 3],
        "ACP_persistence": arr[:, 4],
        "pulse_index": arr[:, 5],
        "maturation_ratio": arr[:, 6],
    }


def _analyze_sobol(
    problem: Dict[str, object],
    y: np.ndarray,
    jobs: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = list(problem["names"])
    if np.allclose(y, y[0], rtol=1e-12, atol=1e-15):
        zeros = np.zeros(len(names), dtype=float)
        return zeros, zeros.copy(), zeros.copy(), zeros.copy()

    parallel = jobs > 1
    try:
        res = sobol.analyze(
            problem,
            y,
            calc_second_order=False,
            num_resamples=1000,
            conf_level=0.95,
            print_to_console=False,
            parallel=parallel,
            n_processors=jobs if parallel else None,
            seed=RNG_SEED,
        )
    except Exception as exc:
        raise RuntimeError("SALib Sobol analysis failed.") from exc
    s1 = np.array(res["S1"], dtype=float)
    s1_conf = np.array(res["S1_conf"], dtype=float)
    st = np.array(res["ST"], dtype=float)
    st_conf = np.array(res["ST_conf"], dtype=float)
    return s1, s1_conf, st, st_conf


def _plot_sobol_stacked_indices(
    names: List[str],
    s1: np.ndarray,
    s1_conf: np.ndarray,
    st: np.ndarray,
    st_conf: np.ndarray,
    title: str,
    out_dir: Path,
    fname: str,
) -> None:
    top_n = min(SOBOL_FIGURE_TOP_N, len(names))
    order = np.argsort(st)[::-1][:top_n]
    labels = [PARAM_DISPLAY.get(names[i], names[i]) for i in order]
    s1_raw = s1[order]
    st_raw = st[order]
    s1_vals = np.clip(s1_raw, 0.0, None)
    s1_err = np.abs(s1_conf[order])
    st_vals = np.clip(st_raw, 0.0, None)
    st_err = np.abs(st_conf[order]) if st_conf.size else np.array([], dtype=float)
    nonadditive_vals = np.clip(st_vals - s1_vals, 0.0, None)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    c_first = "#4C78A8"
    c_nonadditive = "#E69F00"
    edge = "#2F2F2F"
    bar_lw = 0.9
    ax.bar(
        x,
        s1_vals,
        color=c_first,
        edgecolor=edge,
        linewidth=bar_lw,
        label=r"First-order $S_1$",
    )
    ax.bar(
        x,
        nonadditive_vals,
        bottom=s1_vals,
        color=c_nonadditive,
        edgecolor=edge,
        linewidth=bar_lw,
        label=r"Non-additive contribution, $S_T - S_1$",
    )
    ax.errorbar(
        x,
        s1_vals,
        yerr=s1_err,
        fmt="none",
        ecolor="#009E73",
        elinewidth=1.0,
        capsize=3.2,
        capthick=1.0,
        label=r"$S_1$ 95% CI (green)",
        zorder=4,
    )
    if st_err.size:
        st_error = ax.errorbar(
            x,
            st_vals,
            yerr=st_err,
            fmt="none",
            ecolor="#7A3E9D",
            elinewidth=1.0,
            capsize=5.0,
            capthick=1.0,
            label=r"$S_T$ 95% CI (purple, dashed)",
            zorder=5,
        )
        for barlinecol in st_error[2]:
            barlinecol.set_linestyle((0, (2, 2)))
    clean_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", rotation_mode="anchor", fontsize=10.0)
    ax.margins(x=0.045)
    ax.set_ylabel("Sobol sensitivity index")
    ax.set_title(title)
    y_max = 1.0
    if st_err.size:
        y_max = max(y_max, float(np.nanmax(st_vals + st_err)) + 0.04)
    ax.set_ylim(0.0, min(1.15, y_max))
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.tick_params(axis="x", pad=7)
    ax.legend(frameon=False, loc="upper right", fontsize=8.1, handlelength=1.8)
    fig.subplots_adjust(bottom=0.2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    _save(fig, out_dir, fname)


def _write_summary(
    out_dir: Path,
    baseline_params: Dict[str, float],
    baseline_out: ScalarOutputs,
    names: List[str],
    sobol_by_output: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> None:
    lines = [
        "# Sobol Sensitivity Analysis (SALib)",
        "",
        f"Run UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Baseline parameters from main.py defaults",
    ]
    for k in [
        "beta_GP",
        "eta_48h",
        "k_hyd",
        "media_interval",
        "media_renewal_fraction",
        "n",
        "K_A",
        "k_f",
        "k_m",
        "M_max",
        "a_AA0",
        "t_AA_half_life",
        "r_A",
        "r_H",
        "xi_A",
        "xi_H",
    ]:
        if k in baseline_params:
            lines.append(f"- {k}: {baseline_params[k]:.6g}")

    lines.extend(
        [
            f"- Xi_stochasticity: {STOCHASTIC_DEFAULTS.Xi:.6g}",
            f"- r_A_range: [{STOCHASTIC_DEFAULTS.r_A_min:.6g}, {STOCHASTIC_DEFAULTS.r_A_max:.6g}]",
            f"- r_H_range: [{STOCHASTIC_DEFAULTS.r_H_min:.6g}, {STOCHASTIC_DEFAULTS.r_H_max:.6g}]",
        ]
    )

    lines.extend(
        [
            "",
            "## Baseline outputs",
            f"- final apatite H_final: {baseline_out.h_final:.6g}",
            f"- peak ACP A_peak: {baseline_out.a_max:.6g}",
            f"- supersaturation onset time: {baseline_out.t_onset:.6g} days",
            f"- ACP peak time: {baseline_out.t_a_peak:.6g} days",
            f"- ACP persistence: {baseline_out.acp_persistence:.6g}",
            f"- pulse index: {baseline_out.pulse_index:.6g}",
            f"- maturation ratio: {baseline_out.maturation_ratio:.6g}",
        ]
    )

    for output_name, (s1, st) in sobol_by_output.items():
        _, title = OUTPUT_DISPLAY[output_name]
        lines.append("")
        lines.append(f"## Parameters for {title} by total-order ST")
        for i in np.argsort(st)[::-1]:
            lines.append(f"- {names[i]}: S1={s1[i]:.4f}, ST={st[i]:.4f}")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_metadata(out_dir: Path, baseline_params: Dict[str, float], baseline_out: ScalarOutputs) -> None:
    def json_safe(value):
        if isinstance(value, (str, bool)):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    payload = json.dumps({k: json_safe(v) for k, v in baseline_params.items()}, indent=2)
    (out_dir / "baseline_params.json").write_text(payload, encoding="utf-8")
    (out_dir / "baseline_outputs.json").write_text(
        json.dumps(
            {
                "H_final": float(baseline_out.h_final),
                "A_peak": float(baseline_out.a_max),
                "t_onset": float(baseline_out.t_onset),
                "t_A_peak": float(baseline_out.t_a_peak),
                "ACP_persistence": float(baseline_out.acp_persistence),
                "pulse_index": float(baseline_out.pulse_index),
                "maturation_ratio": float(baseline_out.maturation_ratio),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _save_sobol_csv(path: Path, names: List[str], values: np.ndarray) -> None:
    lines = ["parameter,value"]
    lines.extend(f"{name},{float(value):.16g}" for name, value in zip(names, values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_sobol_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    rows = path.read_text(encoding="utf-8").splitlines()[1:]
    names = []
    values = []
    for row in rows:
        if not row.strip():
            continue
        name, value = row.split(",", 1)
        names.append(name.strip())
        values.append(float(value))
    return names, np.array(values, dtype=float)


def _load_sobol_values(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=float)
    _, values = _load_sobol_csv(path)
    return values


def analyze_saved_results(results_dir: Path, figure_formats: tuple[str, ...] = ("pdf", "png")) -> None:
    global FIGURE_FORMATS
    FIGURE_FORMATS = figure_formats
    configure_matplotlib(PLOT_DEFAULTS)
    baseline = json.loads((results_dir / "baseline_params.json").read_text(encoding="utf-8"))
    sobol_loaded = {}
    for output_name, (slug, _) in OUTPUT_DISPLAY.items():
        path = results_dir / f"sobol_s1_{slug}.csv"
        if not path.exists():
            continue
        names, s1 = _load_sobol_csv(path)
        c = _load_sobol_values(results_dir / f"sobol_s1conf_{slug}.csv")
        st = _load_sobol_values(results_dir / f"sobol_st_{slug}.csv")
        stc = _load_sobol_values(results_dir / f"sobol_stconf_{slug}.csv")
        if c.size == 0:
            c = np.zeros_like(s1)
        if st.size == 0:
            st = np.zeros_like(s1)
        sobol_loaded[output_name] = (names, s1, c, st, stc)

    sim_base, _ = _simulate_with_params(baseline, {})
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    ax.plot(sim_base.t, sim_base.A, lw=1.9, color="#1f4e79", label="ACP")
    ax.plot(sim_base.t, sim_base.H, lw=1.9, color="#2a9d8f", label="Apatite")
    clean_axis(ax)
    ax.set_xlim(0, RUN_DEFAULTS.t_end)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Amount (a.u.)")
    ax.set_title("Baseline mineralization dynamics")
    ax.legend(frameon=True, edgecolor="#d9d9d9")
    _save(fig, results_dir, "fig1_baseline")

    selected_outputs = (("H_final", 4), ("A_peak", 5))
    for output_name, idx in selected_outputs:
        if output_name not in sobol_loaded:
            continue
        names, s1, c, st, stc = sobol_loaded[output_name]
        slug, title = OUTPUT_DISPLAY[output_name]
        _plot_sobol_stacked_indices(names, s1, c, st, stc, title, results_dir, f"fig{idx}_top5_{slug}_s1")
    print(f"Analysis figures generated from saved results: {results_dir}")


def main(
    sobol_n_base: int = 1024,
    jobs: int = RUN_DEFAULTS.jobs,
    save_figures: bool = False,
    replicates_per_sample: int = 1,
    figure_formats: tuple[str, ...] = ("pdf", "png"),
) -> None:
    global FIGURE_FORMATS
    FIGURE_FORMATS = figure_formats
    configure_matplotlib(PLOT_DEFAULTS)
    sobol_n_base = _validate_sobol_n_base(sobol_n_base)
    out_dir = create_run_directory("sensitivity", "sobol")

    base = build_base_params()
    base["media_interval"] = RUN_DEFAULTS.media_interval
    baseline = dict(base)
    sim_base, out_base = _simulate_with_params(baseline, {})
    if jobs > 1:
        problem, y = _run_sobol_parallel(baseline, sobol_n_base, jobs, replicates_per_sample=replicates_per_sample)
    else:
        problem, y = _run_sobol(baseline, sobol_n_base, replicates_per_sample=replicates_per_sample)
    names = list(problem["names"])
    sobol_results = {}
    for output_name, values in y.items():
        slug, _ = OUTPUT_DISPLAY[output_name]
        s1, c, st, stc = _analyze_sobol(problem, values, jobs=jobs)
        sobol_results[output_name] = (s1, c, st, stc)
        _save_sobol_csv(out_dir / f"sobol_s1_{slug}.csv", names, s1)
        _save_sobol_csv(out_dir / f"sobol_s1conf_{slug}.csv", names, c)
        _save_sobol_csv(out_dir / f"sobol_st_{slug}.csv", names, st)
        _save_sobol_csv(out_dir / f"sobol_stconf_{slug}.csv", names, stc)
    _save_metadata(out_dir, baseline, out_base)

    _write_summary(out_dir, baseline, out_base, names, {k: (v[0], v[2]) for k, v in sobol_results.items()})
    if save_figures:
        fig, ax = plt.subplots(figsize=(7.0, 4.1))
        ax.plot(sim_base.t, sim_base.A, lw=1.9, color="#1f4e79", label="ACP")
        ax.plot(sim_base.t, sim_base.H, lw=1.9, color="#2a9d8f", label="Apatite")
        clean_axis(ax)
        ax.set_xlim(0, RUN_DEFAULTS.t_end)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Amount (a.u.)")
        ax.set_title("Baseline mineralization dynamics")
        ax.legend(frameon=True, edgecolor="#d9d9d9")
        _save(fig, out_dir, "fig1_baseline")
        selected_outputs = (("H_final", 4), ("A_peak", 5))
        for output_name, idx in selected_outputs:
            s1, c, st, stc = sobol_results[output_name]
            slug, title = OUTPUT_DISPLAY[output_name]
            _plot_sobol_stacked_indices(names, s1, c, st, stc, title, out_dir, f"fig{idx}_top5_{slug}_s1")
    print(f"Sobol SALib pipeline complete. Output: {out_dir}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run Sobol sensitivity with SALib.")
    parser.add_argument(
        "--sobol-n-base",
        type=int,
        default=1024,
        help="Positive power-of-two base N for SALib Sobol sampling; total evaluations are N*(D+2) with second-order indices disabled.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=RUN_DEFAULTS.jobs,
        help="Parallel worker count. Values >1 parallelize model evaluations and SALib Sobol bootstrap analysis.",
    )
    parser.add_argument("--save-figures", action="store_true", help="Generate publication PDF and 600-dpi PNG figures during the run.")
    parser.add_argument("--figure-formats", default="pdf,png", help="Comma-separated: pdf,png; empty skips figures.")
    parser.add_argument(
        "--replicates-per-sample",
        type=int,
        default=1,
        help="Must be 1; Sobol responses are deterministic for each sampled input vector.",
    )
    parser.add_argument("--analyze-only", action="store_true", help="Generate figures from previously saved results without rerunning Sobol simulations.")
    parser.add_argument("--results-dir", type=str, default="", help="Existing results directory for --analyze-only.")
    args = parser.parse_args()
    if args.analyze_only:
        if not args.results_dir:
            raise ValueError("--results-dir is required with --analyze-only")
        analyze_saved_results(Path(args.results_dir), parse_figure_formats(args.figure_formats))
    else:
        main(
            sobol_n_base=args.sobol_n_base,
            jobs=max(1, args.jobs),
            save_figures=args.save_figures,
            replicates_per_sample=max(1, args.replicates_per_sample),
            figure_formats=parse_figure_formats(args.figure_formats),
        )


if __name__ == "__main__":
    cli()
