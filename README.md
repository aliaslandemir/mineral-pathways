# ACP-to-Apatite Model

Reduced-order Python model of amorphous calcium phosphate (ACP)-like precursor
formation and maturation into apatite-like mineral under repeated medium-renewal
cycles.

## Repository Layout

- `main.py` runs the primary stochastic model workflow and exports the main
  figure set.
- `src/` contains the supported model, sampling, plotting, calibration,
  validation, and sensitivity workflows.
- `src/old/` preserves retired exploratory scripts for reference only.
- `output/` is generated at runtime and is ignored by Git.

## Installation

Run commands from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Main Figure Workflow

```bash
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
```

This workflow writes timestamped outputs under `output/main/` and exports:

- `main_fig02_soluble_ion_dynamics`
- `main_fig03_acp_apatite_dynamics`
- `main_fig06_endpoint_parameter_sweep`
- `supp_fig02_formation_maturation_rates`
- `supp_fig03_ascorbate_matrix_support_sweep`

Use `--random-seed 42` for a reproducible stochastic run. Supplying
`--random-seed` without a value generates a fresh stochastic seed.

## Additional Workflows

Protocol prediction heatmap:

```bash
python -m src.protocol_main_prediction_figure \
    --dt 0.05 \
    --figure-formats pdf,png
```

Sobol global sensitivity analysis:

```bash
python -m src.sobol_sensitivity \
    --sobol-n-base 1024 \
    --jobs 6 \
    --save-figures \
    --figure-formats pdf,png
```

ARS external consistency check:

```bash
python -m src.validation_ars \
    --n-runs 100 \
    --t-end 30 \
    --dt 0.05 \
    --media-interval 2 \
    --k-m 0.2 \
    --a-AA0 0.28 \
    --t-AA-half-life 24.0 \
    --figure-formats pdf,png
```

ARS trajectory calibration:

```bash
python -m src.calibration_ars \
    --n-runs 100 \
    --t-end 30 \
    --dt 0.05 \
    --media-interval 2 \
    --n-starts 30 \
    --grid-size 60 \
    --jobs 6 \
    --figure-formats pdf,png
```

Structural-control ablations:

```bash
python -m src.structural_controls --figure-formats pdf,png
```

## Model Notes

AA2P/ascorbate decays after each medium renewal according to:

```text
a_AA(t) = a_AA0 * 2^(-24 * tau / t_AA_half_life)
```

Here `tau` is in days and `t_AA_half_life` is in hours. Matrix support is a
normalized saturating function of the time-dependent ascorbate concentration.
The default half-life is 24 h, and the fixed internal half-support
concentration is 0.14 mM.

All supported workflows write beneath `output/`. Generated outputs, virtual
environments, bytecode caches, and old `results/` artifacts are ignored by Git.
