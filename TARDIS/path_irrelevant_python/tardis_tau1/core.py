from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import Tau1Config
from .matio import load_mat_variable


@dataclass
class PairHistogram:
    histogram: np.ndarray
    candidate_pairs: int
    retained_pairs: int
    zero_distances: int
    source_frame_pairs: int


@dataclass
class CellResult:
    ids: dict[str, str]
    source: dict[str, Any]
    metadata: dict[str, Any]
    parameters: dict[str, Any]
    counts: dict[str, Any]
    distributions: dict[str, Any]
    qc: dict[str, Any]
    schema_version: str


@dataclass
class GenotypeResult:
    ids: dict[str, str]
    cells: dict[str, Any]
    parameters: dict[str, Any]
    distributions: dict[str, Any]
    qc: dict[str, Any]
    bootstrap: dict[str, Any]
    schema_version: str


def analyze_cell(
    input_file: str | Path,
    ids: dict[str, Any] | None = None,
    config: Tau1Config | None = None,
) -> CellResult:
    """Compute tau=1 Total, DANAE-like Inter, and residual Intra for one cell."""
    ids = dict(ids or {})
    config = config or Tau1Config()
    variable_name = str(ids.get("position_variable", "pos"))
    coordinate_unit = str(ids.get("coordinate_unit", "m"))
    positions = load_mat_variable(input_file, variable_name)
    positions = _validate_and_convert_positions(positions, coordinate_unit)
    frame_values, frame_positions, frame_counts = _group_frames(positions)

    edges = np.linspace(0.0, config.max_distance_m, config.n_bins + 1)
    bin_mids = (edges[:-1] + edges[1:]) / 2
    total_stats = pair_histogram(frame_values, frame_positions, edges, "total", config)
    inter_stats = pair_histogram(frame_values, frame_positions, edges, "inter", config)
    if total_stats.retained_pairs == 0:
        raise ValueError("No tau=1 distances were retained")
    if inter_stats.retained_pairs == 0:
        raise ValueError("No same-frame Inter distances were retained")

    total = total_stats.histogram / total_stats.histogram.sum()
    inter_conditional = inter_stats.histogram / inter_stats.histogram.sum()
    tail_start = int(np.floor(config.min_distance_background_fit * config.n_bins))
    background_tail = inter_conditional[tail_start:]
    tail_denominator = float(background_tail @ background_tail)
    if tail_denominator <= 0:
        raise ValueError("The Inter tail has zero norm")
    beta = float(background_tail @ total[tail_start:] / tail_denominator)
    inter_contribution = beta * inter_conditional
    intra_contribution = total - inter_contribution
    intra_mass = 1.0 - beta
    intra_conditional = (
        intra_contribution / intra_mass
        if abs(intra_mass) > 100 * np.finfo(float).eps
        else np.full_like(intra_contribution, np.nan)
    )

    reconstruction_error = float(
        np.max(np.abs(total - inter_contribution - intra_contribution))
    )
    path = Path(input_file).resolve()
    stat = path.stat()
    missing_frames = int(frame_values[-1] - frame_values[0] + 1 - frame_values.size)
    duplicate_rows = int(positions.shape[0] - np.unique(positions, axis=0).shape[0])
    extra_zero_distances = max(0, inter_stats.zero_distances - positions.shape[0])
    flags: list[str] = []
    if not 0 <= beta <= 1:
        flags.append("beta_out_of_range")
    if np.any(intra_contribution < 0):
        flags.append("negative_intra_bins")
    if missing_frames:
        flags.append("missing_frames")
    if extra_zero_distances:
        flags.append("coincident_same_frame_detections_removed")
    if inter_stats.histogram[tail_start:].sum() < 100:
        flags.append("sparse_inter_tail")
    if reconstruction_error > 1e-12:
        flags.append("reconstruction_error")

    canonical_ids = {
        "condition": str(ids.get("condition", "UNASSIGNED")),
        "target": str(ids.get("target", "UNASSIGNED")),
        "genotype": str(ids.get("genotype", "UNASSIGNED")),
        "replicate": str(ids.get("replicate", "UNASSIGNED")),
        "cell": str(ids.get("cell", path.stem)),
    }
    frame_time = _optional_float(ids.get("frame_time_s"))
    return CellResult(
        ids=canonical_ids,
        source={
            "input_file": str(path),
            "position_variable": variable_name,
            "file_bytes": stat.st_size,
            "file_mtime_ns": stat.st_mtime_ns,
        },
        metadata={
            "frame_time_s": frame_time,
            "tau_frames": 1,
            "lag_seconds": frame_time,
            "input_coordinate_unit": coordinate_unit,
            "analysis_coordinate_unit": "m",
        },
        parameters={
            **config.to_dict(),
            "tail_start_bin_one_based": tail_start + 1,
            "edges_m": edges,
            "bin_mids_m": bin_mids,
        },
        counts={
            "localizations": int(positions.shape[0]),
            "frames": int(frame_values.size),
            "frame_min": int(frame_values[0]),
            "frame_max": int(frame_values[-1]),
            "missing_frames": missing_frames,
            "duplicate_rows": duplicate_rows,
            "per_occupied_frame": frame_counts,
            "total_candidate_pairs": total_stats.candidate_pairs,
            "total_retained_pairs": total_stats.retained_pairs,
            "total_zero_distances": total_stats.zero_distances,
            "total_histogram": total_stats.histogram,
            "inter_candidate_pairs": inter_stats.candidate_pairs,
            "inter_zero_distances_removed": inter_stats.zero_distances,
            "inter_retained_pairs": inter_stats.retained_pairs,
            "inter_histogram": inter_stats.histogram,
        },
        distributions={
            "total": total,
            "inter_contribution": inter_contribution,
            "intra_contribution": intra_contribution,
            "inter_conditional": inter_conditional,
            "intra_conditional": intra_conditional,
            "beta": beta,
        },
        qc={
            "pass": True,
            "status": "pass" if not flags else "pass_with_warnings",
            "flags": flags,
            "metrics": {
                "reconstruction_error": reconstruction_error,
                "normalization_error": float(abs(total.sum() - 1)),
                "negative_intra_bins": int(np.count_nonzero(intra_contribution < 0)),
                "negative_intra_mass": float(-np.minimum(intra_contribution, 0).sum()),
                "minimum_intra_contribution": float(intra_contribution.min()),
                "extra_zero_distances_removed": int(extra_zero_distances),
                "total_retained_fraction": total_stats.retained_pairs
                / max(total_stats.candidate_pairs, 1),
                "inter_retained_fraction": inter_stats.retained_pairs
                / max(inter_stats.candidate_pairs, 1),
            },
        },
        schema_version=config.schema_version,
    )


def pair_histogram(
    frame_values: np.ndarray,
    frame_positions: list[np.ndarray],
    edges: np.ndarray,
    mode: str,
    config: Tau1Config,
) -> PairHistogram:
    """Accumulate cross-frame or ordered same-frame distance counts."""
    if mode not in {"total", "inter"}:
        raise ValueError('mode must be "total" or "inter"')
    histogram = np.zeros(edges.size - 1, dtype=np.int64)
    candidate_pairs = retained_pairs = zero_distances = source_frame_pairs = 0
    frame_index = {int(frame): index for index, frame in enumerate(frame_values)}
    for index, frame in enumerate(frame_values):
        if mode == "total":
            other_index = frame_index.get(int(frame) + config.tau_frames)
            if other_index is None:
                continue
            other = frame_positions[other_index]
        else:
            other = frame_positions[index]
        source = frame_positions[index]
        candidate_pairs += source.shape[0] * other.shape[0]
        source_frame_pairs += 1
        for start in range(0, source.shape[0], config.chunk_size):
            block = source[start : start + config.chunk_size]
            dx = block[:, 0, None] - other[:, 0]
            dy = block[:, 1, None] - other[:, 1]
            distances = np.sqrt(dx * dx + dy * dy)
            zero_distances += int(np.count_nonzero(distances == 0))
            if mode == "inter":
                kept = distances[
                    (distances > 0) & (distances <= config.max_distance_m)
                ]
            else:
                kept = distances[distances <= config.max_distance_m]
            retained_pairs += kept.size
            histogram += np.histogram(kept, bins=edges)[0]
    return PairHistogram(
        histogram=histogram,
        candidate_pairs=int(candidate_pairs),
        retained_pairs=int(retained_pairs),
        zero_distances=int(zero_distances),
        source_frame_pairs=int(source_frame_pairs),
    )


def aggregate_genotype(
    cell_results: Iterable[CellResult], config: Tau1Config | None = None
) -> GenotypeResult:
    """Aggregate complete cell contributions with equal cell weight."""
    cells = list(cell_results)
    if not cells:
        raise ValueError("At least one cell result is required")
    config = config or Tau1Config()
    reference_edges = cells[0].parameters["edges_m"]
    for cell in cells:
        if not cell.qc["pass"]:
            raise ValueError(f"QC-failing cell included: {cell.ids['cell']}")
        if not np.array_equal(cell.parameters["edges_m"], reference_edges):
            raise ValueError("Cells in one genotype must use identical bins")

    total = np.mean([cell.distributions["total"] for cell in cells], axis=0)
    inter = np.mean(
        [cell.distributions["inter_contribution"] for cell in cells], axis=0
    )
    intra = np.mean(
        [cell.distributions["intra_contribution"] for cell in cells], axis=0
    )
    beta = float(inter.sum())
    inter_conditional = (
        inter / beta
        if abs(beta) > 100 * np.finfo(float).eps
        else np.full_like(inter, np.nan)
    )
    intra_conditional = (
        intra / (1 - beta)
        if abs(1 - beta) > 100 * np.finfo(float).eps
        else np.full_like(intra, np.nan)
    )
    reconstruction_error = float(np.max(np.abs(total - inter - intra)))
    flags: list[str] = []
    if not 0 <= beta <= 1:
        flags.append("beta_out_of_range")
    if np.any(intra < 0):
        flags.append("negative_intra_bins")
    if len(cells) == 1:
        flags.append("single_cell_no_biological_interval")

    first = cells[0]
    return GenotypeResult(
        ids={key: first.ids[key] for key in ("condition", "target", "genotype")},
        cells={
            "included_cell_ids": [cell.ids["cell"] for cell in cells],
            "replicate_ids": [cell.ids["replicate"] for cell in cells],
            "included_count": len(cells),
            "excluded_count": 0,
        },
        parameters=dict(first.parameters),
        distributions={
            "total": total,
            "inter_contribution": inter,
            "intra_contribution": intra,
            "inter_conditional": inter_conditional,
            "intra_conditional": intra_conditional,
            "beta": beta,
        },
        qc={
            "pass": True,
            "status": "pass" if not flags else "pass_with_warnings",
            "flags": flags,
            "metrics": {
                "reconstruction_error": reconstruction_error,
                "negative_intra_bins": int(np.count_nonzero(intra < 0)),
                "negative_intra_mass": float(-np.minimum(intra, 0).sum()),
            },
        },
        bootstrap=_bootstrap_genotype(cells, config),
        schema_version=config.schema_version,
    )


def _bootstrap_genotype(cells: list[CellResult], config: Tau1Config) -> dict[str, Any]:
    n_bins = cells[0].distributions["total"].size
    if len(cells) < 2 or config.bootstrap_count == 0:
        empty = np.full((3, n_bins), np.nan)
        return {
            "status": "single_cell_no_interval",
            "count": config.bootstrap_count,
            "seed": config.bootstrap_seed,
            "percentiles": [2.5, 50.0, 97.5],
            "total": empty.copy(),
            "inter_contribution": empty.copy(),
            "intra_contribution": empty.copy(),
            "beta": np.full(3, np.nan),
        }
    rng = np.random.default_rng(config.bootstrap_seed)
    replicates: dict[str, list[int]] = {}
    for index, cell in enumerate(cells):
        replicates.setdefault(cell.ids["replicate"], []).append(index)
    boot_total = np.zeros((config.bootstrap_count, n_bins))
    boot_inter = np.zeros_like(boot_total)
    boot_intra = np.zeros_like(boot_total)
    boot_beta = np.zeros(config.bootstrap_count)
    for draw in range(config.bootstrap_count):
        selected: list[int] = []
        for indices in replicates.values():
            selected.extend(
                rng.choice(indices, size=len(indices), replace=True).tolist()
            )
        boot_total[draw] = np.mean(
            [cells[i].distributions["total"] for i in selected], axis=0
        )
        boot_inter[draw] = np.mean(
            [cells[i].distributions["inter_contribution"] for i in selected],
            axis=0,
        )
        boot_intra[draw] = np.mean(
            [cells[i].distributions["intra_contribution"] for i in selected],
            axis=0,
        )
        boot_beta[draw] = boot_inter[draw].sum()
    percentiles = [2.5, 50.0, 97.5]
    return {
        "status": "cell_bootstrap",
        "count": config.bootstrap_count,
        "seed": config.bootstrap_seed,
        "percentiles": percentiles,
        "total": np.percentile(boot_total, percentiles, axis=0),
        "inter_contribution": np.percentile(boot_inter, percentiles, axis=0),
        "intra_contribution": np.percentile(boot_intra, percentiles, axis=0),
        "beta": np.percentile(boot_beta, percentiles),
    }


def _validate_and_convert_positions(array: np.ndarray, unit: str) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
        raise ValueError("Position data must be a nonempty numeric N-by-3 matrix")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Position data must contain only finite numeric values")
    positions = array.astype(float, copy=True)
    if not np.array_equal(positions[:, 0], np.round(positions[:, 0])):
        raise ValueError("Frame identifiers must be integers")
    factors = {"m": 1.0, "um": 1e-6, "µm": 1e-6, "nm": 1e-9}
    try:
        factor = factors[unit.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported coordinate unit: {unit}") from exc
    positions[:, 1:] *= factor
    order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    return positions[order]


def _group_frames(
    positions: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    frame_values, starts, counts = np.unique(
        positions[:, 0].astype(np.int64), return_index=True, return_counts=True
    )
    grouped = [
        positions[start : start + count, 1:3]
        for start, count in zip(starts, counts)
    ]
    return frame_values, grouped, counts


def _optional_float(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "nan"}:
        return None
    number = float(value)
    return number if np.isfinite(number) else None
