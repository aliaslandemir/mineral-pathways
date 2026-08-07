from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .model import SimulationResult, simulate, synchronize_hydrolysis_parameters, transient_metrics
from .sampling import apply_stochastic_variability


@dataclass
class StochasticSweepResult:
    representative: List[Tuple[float, SimulationResult, Dict[str, float]]]
    trajectories: Dict[str, Dict[float, np.ndarray]]


def _simulate_sweep_task(task):
    level, index, params, fields, t_end, dt, media_interval, keep_simulation = task
    sim = simulate(params, t_end=t_end, dt=dt, media_interval=media_interval)
    arrays = tuple(np.asarray(getattr(sim, field), dtype=float) for field in fields)
    return level, index, sim if keep_simulation else None, arrays


def _set_sweep_parameter(params: Dict[str, float], name: str, value: float, dt: float, media_interval: float) -> None:
    params[name] = float(value)
    if name == "eta_48h":
        params.pop("_k_hyd_explicit", None)
        params.pop("_eta_dt_explicit", None)
        synchronize_hydrolysis_parameters(params, dt, media_interval)


def run_stochastic_parameter_sweep(
    base_params: Dict[str, float],
    name: str,
    values: Iterable[float],
    samples: Sequence[Dict[str, float]],
    fields: Sequence[str],
    t_end: float,
    dt: float,
    media_interval: float,
    baseline_runs: Sequence[SimulationResult] | None = None,
    jobs: int = 1,
) -> StochasticSweepResult:
    levels = [float(value) for value in values]
    fields = tuple(fields)
    n_samples = len(samples)
    trajectories = {field: {level: [None] * n_samples for level in levels} for field in fields}
    representative_by_level: Dict[float, SimulationResult] = {}
    baseline_value = float(base_params[name])
    tasks = []

    for level in levels:
        reuse_baseline = baseline_runs is not None and np.isclose(level, baseline_value, rtol=1e-12, atol=1e-14)
        for index, sample in enumerate(samples):
            if reuse_baseline:
                sim = baseline_runs[index]
                for field in fields:
                    trajectories[field][level][index] = np.asarray(getattr(sim, field), dtype=float)
                if index == 0:
                    representative_by_level[level] = sim
            else:
                params = apply_stochastic_variability(base_params, sample)
                _set_sweep_parameter(params, name, level, dt, media_interval)
                tasks.append((level, index, params, fields, t_end, dt, media_interval, index == 0))

    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = executor.map(_simulate_sweep_task, tasks, chunksize=max(1, len(tasks) // (jobs * 8)))
            for level, index, representative_sim, arrays in results:
                if representative_sim is not None:
                    representative_by_level[level] = representative_sim
                for field, array in zip(fields, arrays):
                    trajectories[field][level][index] = array
    else:
        for task in tasks:
            level, index, representative_sim, arrays = _simulate_sweep_task(task)
            if representative_sim is not None:
                representative_by_level[level] = representative_sim
            for field, array in zip(fields, arrays):
                trajectories[field][level][index] = array

    representative = []
    for level in levels:
        for field in fields:
            trajectories[field][level] = np.asarray(trajectories[field][level], dtype=float)
        first_sim = representative_by_level[level]
        representative.append((level, first_sim, transient_metrics(first_sim)))

    return StochasticSweepResult(representative=representative, trajectories=trajectories)


def run_parameter_sweep(
    base_params: Dict[str, float],
    name: str,
    values: Iterable[float],
    t_end: float,
    dt: float,
    media_interval: float,
) -> List[Tuple[float, object, Dict[str, float]]]:
    out = []
    for value in values:
        p = dict(base_params)
        _set_sweep_parameter(p, name, float(value), dt, media_interval)
        sim = simulate(p, t_end=t_end, dt=dt, media_interval=media_interval)
        out.append((float(value), sim, transient_metrics(sim)))
    return out


def compare_conditions(
    base_params: Dict[str, float],
    conditions: List[Dict[str, float]],
    t_end: float,
    dt: float,
    media_interval: float,
):
    out = []
    for cond in conditions:
        p = dict(base_params)
        p.update(cond)
        sim = simulate(p, t_end=t_end, dt=dt, media_interval=media_interval)
        out.append((cond, sim, transient_metrics(sim)))
    return out
