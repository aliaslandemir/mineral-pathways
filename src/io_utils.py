from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "output"
SUPPORTED_FIGURE_FORMATS = ("pdf", "png")


def parse_figure_formats(value: str | Iterable[str]) -> tuple[str, ...]:
    raw = value.split(",") if isinstance(value, str) else value
    formats = tuple(dict.fromkeys(str(item).strip().lower() for item in raw if str(item).strip()))
    invalid = sorted(set(formats) - set(SUPPORTED_FIGURE_FORMATS))
    if invalid:
        raise ValueError(f"Unsupported figure format(s): {', '.join(invalid)}. Use pdf and/or png.")
    return formats


def create_run_directory(workflow: str, prefix: str = "run") -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_ROOT / workflow / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def figure_format_dir(run_dir: Path, suffix: str) -> Path:
    suffix = suffix.lower()
    if suffix not in SUPPORTED_FIGURE_FORMATS:
        raise ValueError(f"Unsupported figure format: {suffix}")
    return Path(run_dir) / "figures" / suffix


def figure_path(run_dir: Path, stem: str, suffix: str) -> Path:
    return figure_format_dir(run_dir, suffix) / f"{stem}.{suffix}"


def save_figure(
    fig: plt.Figure,
    run_dir: Path,
    stem: str,
    formats: str | Iterable[str] = SUPPORTED_FIGURE_FORMATS,
    *,
    dpi: int = 300,
    close: bool = True,
    **savefig_kwargs,
) -> tuple[Path, ...]:
    requested = parse_figure_formats(formats)
    saved: list[Path] = []
    for suffix in requested:
        path = figure_path(run_dir, stem, suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        kwargs = {"bbox_inches": "tight", "facecolor": "white", **savefig_kwargs}
        if suffix == "png":
            kwargs.setdefault("dpi", dpi)
        fig.savefig(path, **kwargs)
        saved.append(path)
    if close:
        plt.close(fig)
    return tuple(saved)


def save_figure_to_directory(
    fig: plt.Figure,
    figure_dir: Path,
    stem: str,
    formats: str | Iterable[str] = SUPPORTED_FIGURE_FORMATS,
    *,
    dpi: int = 300,
    close: bool = True,
    **savefig_kwargs,
) -> tuple[Path, ...]:
    requested = parse_figure_formats(formats)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for suffix in requested:
        path = figure_dir / f"{stem}.{suffix}"
        kwargs = {"bbox_inches": "tight", "facecolor": "white", **savefig_kwargs}
        if suffix == "png":
            kwargs.setdefault("dpi", dpi)
        fig.savefig(path, **kwargs)
        saved.append(path)
    if close:
        plt.close(fig)
    return tuple(saved)
