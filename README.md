# ACP-to-HAP Model

Competing-pathway model for osteogenic culture mineralization. The code
separates matrix-associated mineralization from solution-driven calcium
phosphate deposition and generates the analysis tables and manuscript figures.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Default Run

Run from the repository root:

```bash
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
```

The same values are the defaults:

```bash
python main.py
```

`--K_H` is still accepted as a command-line alias for `--K_Q`.

The workflow writes to `output/`.

## Endpoint Equifinality

Run the day-28 endpoint-equifinality search after the main workflow:

```bash
python -m src.endpoint_equifinality_day28 \
  --output-dir output/extrafolder \
  --rerun-search \
  --n-samples 2000 \
  --n-refine 50 \
  --tolerance 0.02 \
  --formats pdf,png \
  --jobs 6
```

## Outputs

```text
output/
├── run_manifest.json
├── model/
├── analysis/
└── figures/
```

Generated outputs are ignored by git.

## Main Files

- `main.py`: end-to-end workflow driver.
- `src/model.py`: model states, rate laws, solver, and ensemble simulation.
- `src/analysis.py`: summary metrics, conservation checks, and parameter diagnostics.
- `src/figures.py`: manuscript figure generation.
- `src/endpoint_equifinality_day28.py`: day-28 endpoint-equifinality search and figure.

## Quick Check

```bash
python main.py --stages model,analysis --output-dir output/check
```
