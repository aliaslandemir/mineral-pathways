"""

python main.py \
  --stages model,analysis,figures \
  --output-dir output \
  --figures 2-11 \
  --formats pdf,png \
  --grid-size 50 \
  --sobol-base 1024 \
  --jobs 6 \
  --n-runs 100 \
  --seed 42 \
  --t-end 28 \
  --dt 0.05 \
  --media-interval 2 \
  --solver-method RK45 \
  --solver-rtol 1e-6 \
  --solver-atol 1e-9 \
  --C_ext 2.00 \
  --P_ext 1.25 \
  --beta_GP 10.0 \
  --eta_48h 0.30 \
  --calcium-exponent 3 \
  --phosphate-exponent 2 \
  --K_A 33.0 \
  --K_Q 66.0 \
  --k_f 2e-1 \
  --M_max 1.0 \
  --k_m 2e-1 \
  --k_b 2e-1 \
  --Q_max 1.0 \
  --n_b 1.0 \
  --r_A_min 1.00 \
  --r_A_max 1.49 \
  --r_H_min 1.50 \
  --r_H_max 1.67

"""




from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from src.analysis import save_analysis, save_ensemble_analysis, save_kq_ka_ratio_sweep
from src.figures import generate_figures, parse_figures
from src.model import (
    DEFAULT_OUTPUT_DIR,
    FIXED_MODEL_ASSUMPTIONS,
    NEW_MODEL_DIR,
    MineralizationModel,
    add_model_arguments,
    add_stochastic_arguments,
    parameters_from_args,
    save_ensemble,
    settings_from_args,
    simulate_ensemble,
    stochastic_from_args,
)


VALID_STAGES = ("model", "analysis", "figures")


def parse_stages(text: str) -> list[str]:
    stages = [value.strip().lower() for value in text.split(",") if value.strip()]
    invalid = set(stages).difference(VALID_STAGES)
    if invalid or not stages:
        raise argparse.ArgumentTypeError(f"Stages must be selected from {', '.join(VALID_STAGES)}")
    return list(dict.fromkeys(stages))


def output_path(path: Path) -> Path:
    return path if path.is_absolute() else NEW_MODEL_DIR / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the complete competing-pathway model, analysis, and figure workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stages", type=parse_stages, default=list(VALID_STAGES))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures", type=parse_figures, default=list(range(2, 11)))
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--grid-size", type=int, default=50)
    parser.add_argument("--sobol-base", type=int, default=1024)
    parser.add_argument("--jobs", type=int, default=6)
    add_model_arguments(parser)
    add_stochastic_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.grid_size < 3 or args.sobol_base < 2 or args.jobs < 1:
        raise ValueError("grid-size >= 3, sobol-base >= 2, and jobs >= 1 are required")
    output_dir = output_path(args.output_dir)
    model_dir = output_dir / "model"
    analysis_dir = output_dir / "analysis"
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    stochastic = stochastic_from_args(args)
    parameters = parameters_from_args(args, stochastic)
    settings = settings_from_args(args)
    result = MineralizationModel(parameters).simulate(settings)
    runs = simulate_ensemble(parameters, settings, stochastic)
    produced: dict[str, str] = {}

    if "model" in args.stages:
        trajectory_path = model_dir / "baseline_trajectory.csv"
        result.to_csv(trajectory_path)
        save_ensemble(runs, model_dir)
        produced["model_trajectory"] = str(trajectory_path)
        print(f"Model: saved {trajectory_path}")

    if "analysis" in args.stages:
        save_analysis(result, analysis_dir)
        save_kq_ka_ratio_sweep(parameters, settings, analysis_dir)
        save_ensemble_analysis(runs, analysis_dir)
        produced["analysis_directory"] = str(analysis_dir)
        print(f"Analysis: saved detailed results to {analysis_dir}")

    if "figures" in args.stages:
        formats = tuple(value.strip().lower() for value in args.formats.split(",") if value.strip())
        generate_figures(args.figures, figures_dir, formats, parameters, settings, args.grid_size, args.sobol_base, args.jobs, args.seed, runs)
        produced["figures_directory"] = str(figures_dir)

    manifest = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "stages": args.stages,
        "model_parameters": asdict(parameters),
        "fixed_model_assumptions": FIXED_MODEL_ASSUMPTIONS,
        "simulation_settings": asdict(settings),
        "stochastic_settings": asdict(stochastic),
        "figure_settings": {"figures": args.figures, "formats": args.formats, "grid_size": args.grid_size, "sobol_base": args.sobol_base, "jobs": args.jobs, "seed": args.seed},
        "outputs": produced,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Workflow complete. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
