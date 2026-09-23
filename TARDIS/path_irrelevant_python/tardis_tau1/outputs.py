from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import Tau1Config
from .core import CellResult, GenotypeResult


def save_result(
    result: CellResult | GenotypeResult,
    output_dir: str | Path,
    config: Tau1Config,
    level: str,
) -> None:
    """Save portable arrays, metadata, a table, and an optional diagnostic plot."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    distributions = result.distributions
    np.savez_compressed(
        output_dir / f"{level}_result.npz",
        edges_m=np.asarray(result.parameters["edges_m"]),
        bin_mids_m=np.asarray(result.parameters["bin_mids_m"]),
        total=np.asarray(distributions["total"]),
        inter_contribution=np.asarray(distributions["inter_contribution"]),
        intra_contribution=np.asarray(distributions["intra_contribution"]),
        inter_conditional=np.asarray(distributions["inter_conditional"]),
        intra_conditional=np.asarray(distributions["intra_conditional"]),
        beta=np.asarray(distributions["beta"]),
    )
    metadata = asdict(result)
    for key in ("parameters", "counts", "distributions", "bootstrap"):
        if key in metadata:
            metadata[key] = _json_safe(metadata[key])
    (output_dir / f"{level}_result.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_distribution_csv(result, output_dir / "distributions.csv")
    if config.save_plots:
        _write_diagnostic_plot(result, output_dir / "diagnostic.png", level)


def safe_component(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))
    return cleaned or "UNASSIGNED"


def _write_distribution_csv(
    result: CellResult | GenotypeResult, path: Path
) -> None:
    d = result.distributions
    x = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "distance_um",
                "total",
                "inter_contribution",
                "intra_contribution",
                "inter_conditional",
                "intra_conditional",
            ]
        )
        writer.writerows(
            zip(
                x,
                d["total"],
                d["inter_contribution"],
                d["intra_contribution"],
                d["inter_conditional"],
                d["intra_conditional"],
            )
        )


def _write_diagnostic_plot(
    result: CellResult | GenotypeResult, path: Path, level: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = result.distributions
    x = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    ax.plot(x, d["total"], color="#2a78d6", linewidth=2.0, label="Total")
    ax.plot(
        x,
        d["inter_contribution"],
        color="#eb6834",
        linewidth=2.0,
        linestyle="--",
        label="Inter contribution",
    )
    ax.plot(
        x,
        d["intra_contribution"],
        color="#1baf7a",
        linewidth=2.0,
        linestyle="-.",
        label="Intra contribution",
    )
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
    ax.set(
        xlabel="Distance (µm)",
        ylabel="Probability contribution",
        title=f"{level.capitalize()} τ=1 distribution · β={d['beta']:.4f}",
    )
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path, dpi=180, facecolor="white")
    plt.close(fig)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
