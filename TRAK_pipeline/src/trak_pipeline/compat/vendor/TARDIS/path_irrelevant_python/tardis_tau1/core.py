from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np

from .config import Tau1Config
from .matio import load_mat_variable, load_v73_base_records, load_v73_cell_record
from .provenance import collect_environment_provenance, fingerprint_file


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
    summaries: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenotypeResult:
    """Legacy equal-cell result retained for version-1 batch compatibility."""

    ids: dict[str, str]
    cells: dict[str, Any]
    parameters: dict[str, Any]
    distributions: dict[str, Any]
    qc: dict[str, Any]
    bootstrap: dict[str, Any]
    schema_version: str


@dataclass
class ReplicaResult:
    ids: dict[str, str]
    source: dict[str, Any]
    cells: dict[str, Any]
    parameters: dict[str, Any]
    counts: dict[str, Any]
    distributions: dict[str, Any]
    qc: dict[str, Any]
    schema_version: str
    summaries: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def analyze_cell(
    input_file: str | Path,
    ids: dict[str, Any] | None = None,
    config: Tau1Config | None = None,
) -> CellResult:
    """Compute tau=1 Total, DANAE-like Inter, and residual Intra for one cell."""
    started = time.perf_counter()
    ids = dict(ids or {})
    config = config or Tau1Config()
    path = Path(input_file).resolve()
    fingerprint = fingerprint_file(path)
    provenance = collect_environment_provenance(config)
    provenance["input"] = {"path": str(path), **fingerprint}
    variable_name = str(ids.get("position_variable", "pos"))
    coordinate_unit = str(ids.get("coordinate_unit", "m"))
    frame_time = _optional_float(ids.get("frame_time_s"))
    record_name = ids.get("record_name")
    import_metadata: dict[str, Any] = {}
    if record_name is not None:
        if frame_time is None or frame_time <= 0:
            raise ValueError("frame_time_s is required for MATLAB v7.3 cell records")
        pixel_size_m = _required_positive_float(ids.get("pixel_size_m"), "pixel_size_m")
        positions, import_metadata = load_v73_cell_record(
            path,
            str(record_name),
            frame_interval_s=frame_time,
            pixel_size_m=pixel_size_m,
            variable_name=variable_name,
        )
        coordinate_unit = "m"
    else:
        positions = load_mat_variable(path, variable_name)
    stat = path.stat()
    source = {
        "input_file": str(path),
        "position_variable": variable_name,
        "file_bytes": stat.st_size,
        "file_mtime_ns": stat.st_mtime_ns,
        "file_sha256": fingerprint["sha256"],
    }
    result = analyze_positions(
        positions,
        ids=ids,
        source=source,
        metadata=import_metadata,
        coordinate_unit=coordinate_unit,
        frame_time_s=frame_time,
        config=config,
        provenance=provenance,
    )
    result.provenance["execution"] = {
        "analysis_runtime_seconds": time.perf_counter() - started,
        "runtime_scope": "input hashing, MAT loading, and cell analysis; output serialization excluded",
    }
    return result


def analyze_positions(
    positions: np.ndarray,
    *,
    ids: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    coordinate_unit: str = "m",
    frame_time_s: float | None = None,
    config: Tau1Config | None = None,
    provenance: dict[str, Any] | None = None,
) -> CellResult:
    """Analyze one cell from an in-memory N-by-3 frame/x/y array."""
    started = time.perf_counter()
    ids = dict(ids or {})
    source = dict(source or {})
    metadata = dict(metadata or {})
    config = config or Tau1Config()
    provenance = deepcopy(provenance) if provenance is not None else collect_environment_provenance(config)
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
    background = inter_stats.histogram / inter_stats.histogram.sum()
    decomposition = decompose_distribution(total, background, config)
    distributions = add_distribution_densities(
        {"total": total, "inter_conditional": background, **decomposition}, edges
    )
    intra = decomposition["intra_contribution"]
    positive_intra = summarize_positive_intra(intra, edges)
    tail_start = int(np.floor(config.min_distance_background_fit * config.n_bins))
    reconstruction_error = float(
        np.max(np.abs(total - decomposition["inter_contribution"] - intra))
    )
    missing_frames = int(frame_values[-1] - frame_values[0] + 1 - frame_values.size)
    duplicate_rows = int(positions.shape[0] - np.unique(positions, axis=0).shape[0])
    extra_zero_distances = max(0, inter_stats.zero_distances - positions.shape[0])
    flags: list[str] = []
    if decomposition["beta_was_clipped"]:
        flags.append("beta_clipped")
    if np.any(intra < 0):
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
        "replicate": str(ids.get("replicate", "UNASSIGNED")),
        "cell": str(ids.get("cell", metadata.get("record_name", "UNASSIGNED"))),
    }
    if "genotype" in ids:
        canonical_ids["genotype"] = str(ids["genotype"])
    return CellResult(
        ids=canonical_ids,
        source=source,
        metadata={
            "frame_time_s": frame_time_s,
            "tau_frames": config.tau_frames,
            "lag_seconds": None if frame_time_s is None else frame_time_s * config.tau_frames,
            "input_coordinate_unit": metadata.get("input_coordinate_unit", coordinate_unit),
            "analysis_coordinate_unit": "m",
            **metadata,
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
            "total_source_frame_pairs": total_stats.source_frame_pairs,
            "total_candidate_pairs": total_stats.candidate_pairs,
            "total_retained_pairs": total_stats.retained_pairs,
            "total_zero_distances": total_stats.zero_distances,
            "total_histogram": total_stats.histogram,
            "inter_source_frames": inter_stats.source_frame_pairs,
            "inter_candidate_pairs": inter_stats.candidate_pairs,
            "inter_zero_distances_removed": inter_stats.zero_distances,
            "inter_retained_pairs": inter_stats.retained_pairs,
            "inter_histogram": inter_stats.histogram,
        },
        distributions=distributions,
        summaries={"positive_intra": positive_intra},
        qc={
            "pass": True,
            "status": "pass" if not flags else "pass_with_warnings",
            "flags": flags,
            "metrics": {
                "reconstruction_error": reconstruction_error,
                "normalization_error": float(abs(total.sum() - 1)),
                "negative_intra_bins": int(np.count_nonzero(intra < 0)),
                "negative_intra_mass": float(-np.minimum(intra, 0).sum()),
                "minimum_intra_contribution": float(intra.min()),
                "extra_zero_distances_removed": int(extra_zero_distances),
                "total_retained_fraction": total_stats.retained_pairs
                / max(total_stats.candidate_pairs, 1),
                "inter_retained_fraction": inter_stats.retained_pairs
                / max(inter_stats.candidate_pairs, 1),
            },
        },
        provenance={
            **provenance,
            "execution": {
                "analysis_runtime_seconds": time.perf_counter() - started,
                "runtime_scope": "in-memory cell analysis; output serialization excluded",
            },
        },
        schema_version=config.schema_version,
    )


def analyze_replica(
    input_file: str | Path,
    *,
    condition: str,
    target: str,
    replica: str | None = None,
    frame_interval_s: float = 0.02,
    pixel_size_m: float = 117e-9,
    variable_name: str = "data",
    config: Tau1Config | None = None,
    fingerprint: dict[str, Any] | None = None,
) -> tuple[ReplicaResult, list[CellResult]]:
    """Analyze every valid Base cell in one v7.3 MAT replica."""
    started = time.perf_counter()
    config = config or Tau1Config()
    path = Path(input_file).resolve()
    if fingerprint is None:
        fingerprint = fingerprint_file(path)
    else:
        fingerprint = dict(fingerprint)
        digest = fingerprint.get("sha256")
        size = fingerprint.get("file_bytes")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in digest)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError("Invalid precomputed input fingerprint")
    shared_provenance = collect_environment_provenance(config)
    shared_provenance["input"] = {"path": str(path), **fingerprint}
    replica_id = replica or path.stem
    records, discovered, failures = load_v73_base_records(
        path,
        frame_interval_s=frame_interval_s,
        pixel_size_m=pixel_size_m,
        variable_name=variable_name,
    )
    stat = path.stat()
    cells: list[CellResult] = []
    for record in records:
        try:
            cells.append(
                analyze_positions(
                    record.positions,
                    ids={
                        "condition": condition,
                        "target": target,
                        "replicate": replica_id,
                        "cell": record.name,
                    },
                    source={
                        "input_file": str(path),
                        "position_variable": variable_name,
                        "record_index": record.record_index,
                        "record_name": record.name,
                        "file_bytes": stat.st_size,
                        "file_mtime_ns": stat.st_mtime_ns,
                        "file_sha256": fingerprint["sha256"],
                    },
                    metadata=record.metadata,
                    coordinate_unit="m",
                    frame_time_s=frame_interval_s,
                    config=config,
                    provenance=shared_provenance,
                )
            )
        except (TypeError, ValueError) as exc:
            failures.append(
                {
                    "cell_id": record.name,
                    "record_index": record.record_index,
                    "error": str(exc),
                }
            )
    if not cells:
        raise ValueError("No valid Base cells were available for replica aggregation")
    result = aggregate_replica(
        cells,
        config=config,
        source={
            "input_file": str(path),
            "position_variable": variable_name,
            "file_bytes": stat.st_size,
            "file_mtime_ns": stat.st_mtime_ns,
            "file_sha256": fingerprint["sha256"],
            "frame_interval_s": frame_interval_s,
            "pixel_size_m": pixel_size_m,
        },
        discovered_cell_ids=discovered,
        excluded_cells=failures,
        provenance=shared_provenance,
    )
    result.provenance["execution"] = {
        "analysis_runtime_seconds": time.perf_counter() - started,
        "runtime_scope": "input hashing, MAT decoding, all cell analyses, and replica aggregation; output serialization excluded",
    }
    return result, cells


def decompose_distribution(
    total: np.ndarray, background: np.ndarray, config: Tau1Config
) -> dict[str, Any]:
    """Fit constrained background mass and return an exact Total decomposition."""
    total = np.asarray(total, dtype=float)
    background = np.asarray(background, dtype=float)
    if total.shape != background.shape or total.ndim != 1:
        raise ValueError("Total and background must be equal-length vectors")
    if not np.isfinite(total).all() or not np.isfinite(background).all():
        raise ValueError("Total and background must contain only finite values")
    tail_start = int(np.floor(config.min_distance_background_fit * total.size))
    background_tail = background[tail_start:]
    total_tail = total[tail_start:]
    numerator = float(background_tail @ total_tail)
    denominator = float(background_tail @ background_tail)
    if denominator <= 0:
        raise ValueError("The Inter tail has zero norm")
    beta_raw = numerator / denominator
    beta = float(np.clip(beta_raw, 0.0, 1.0))
    inter = beta * background
    intra = total - inter
    intra_mass = 1.0 - beta
    intra_conditional = (
        intra / intra_mass
        if abs(intra_mass) > 100 * np.finfo(float).eps
        else np.full_like(intra, np.nan)
    )
    return {
        "inter_contribution": inter,
        "intra_contribution": intra,
        "intra_conditional": intra_conditional,
        "beta": beta,
        "beta_raw": float(beta_raw),
        "beta_was_clipped": bool(beta != beta_raw),
        "tail_fit_numerator": numerator,
        "tail_fit_denominator": denominator,
        "tail_bin_count": int(total.size - tail_start),
    }


def add_distribution_densities(
    distributions: dict[str, Any], edges_m: np.ndarray
) -> dict[str, Any]:
    """Return a distribution mapping augmented with densities in inverse micrometres."""
    edges = np.asarray(edges_m, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.diff(edges) > 0):
        raise ValueError("Distance edges must be a strictly increasing vector")
    widths_um = np.diff(edges) * 1e6
    result = dict(distributions)
    density_keys = {
        "total": "total_density_um_inv",
        "inter_conditional": "inter_conditional_density_um_inv",
        "inter_contribution": "inter_contribution_density_um_inv",
        "intra_contribution": "intra_contribution_density_um_inv",
        "intra_conditional": "intra_conditional_density_um_inv",
        "average_inter_conditional_shape": "average_inter_conditional_density_um_inv",
    }
    for probability_key, density_key in density_keys.items():
        if probability_key not in result:
            continue
        values = np.asarray(result[probability_key], dtype=float)
        if values.ndim != 1 or values.size != widths_um.size:
            raise ValueError(f"{probability_key} must match the distance bins")
        result[density_key] = values / widths_um
    return result


def summarize_positive_intra(
    intra_contribution: np.ndarray,
    edges_m: np.ndarray,
    *,
    window_upper_bound_m: float = 300e-9,
) -> dict[str, Any]:
    """Summarize the positive component without altering canonical signed Intra."""
    intra = np.asarray(intra_contribution, dtype=float)
    edges = np.asarray(edges_m, dtype=float)
    if intra.ndim != 1 or edges.ndim != 1 or edges.size != intra.size + 1:
        raise ValueError("Intra and distance edges must describe the same 1D bins")
    widths_m = np.diff(edges)
    if not np.all(widths_m > 0):
        raise ValueError("Distance edges must be strictly increasing")
    if not np.isfinite(intra).all() or not np.isfinite(edges).all():
        raise ValueError("Intra and distance edges must be finite")
    if window_upper_bound_m < edges[0]:
        raise ValueError("Positive-Intra window must overlap the histogram domain")

    positive = np.maximum(intra, 0.0)
    widths_um = widths_m * 1e6
    density = positive / widths_um
    total_positive = float(positive.sum())
    overlap = np.maximum(
        0.0,
        np.minimum(edges[1:], window_upper_bound_m) - np.maximum(edges[:-1], 0.0),
    )
    window_mass = float(np.sum(positive * overlap / widths_m))

    if total_positive <= 0:
        peak_index = -1
        peak_distance_m = peak_probability = peak_density = float("nan")
        mean_distance_m = median_distance_m = float("nan")
    else:
        peak_index = int(np.argmax(density))
        mids = (edges[:-1] + edges[1:]) / 2
        peak_distance_m = float(mids[peak_index])
        peak_probability = float(positive[peak_index])
        peak_density = float(density[peak_index])
        mean_distance_m = float(np.sum(positive * mids) / total_positive)
        target = 0.5 * total_positive
        cumulative = np.cumsum(positive)
        median_index = int(np.searchsorted(cumulative, target, side="left"))
        prior = float(cumulative[median_index - 1]) if median_index else 0.0
        fraction = (target - prior) / positive[median_index]
        median_distance_m = float(edges[median_index] + fraction * widths_m[median_index])

    return {
        "peak_bin_index_zero_based": peak_index,
        "peak_distance_m": peak_distance_m,
        "peak_distance_um": peak_distance_m * 1e6,
        "peak_probability_contribution": peak_probability,
        "peak_density_um_inv": peak_density,
        "total_positive_mass": total_positive,
        "positive_mass_0_300_nm": window_mass,
        "positive_weighted_mean_distance_m": mean_distance_m,
        "positive_weighted_mean_distance_um": mean_distance_m * 1e6,
        "positive_weighted_median_distance_m": median_distance_m,
        "positive_weighted_median_distance_um": median_distance_m * 1e6,
        "window_upper_bound_m": float(window_upper_bound_m),
    }


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


def aggregate_replica(
    cell_results: Iterable[CellResult],
    config: Tau1Config | None = None,
    *,
    source: dict[str, Any] | None = None,
    discovered_cell_ids: list[str] | None = None,
    excluded_cells: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> ReplicaResult:
    """Build the canonical equal-cell replica decomposition."""
    cells = list(cell_results)
    if not cells:
        raise ValueError("At least one valid cell result is required")
    config = config or Tau1Config()
    first = cells[0]
    edges = np.asarray(first.parameters["edges_m"])
    replica_keys = ("condition", "target", "replicate")
    reference_ids = {key: first.ids[key] for key in replica_keys}
    for cell in cells:
        if not cell.qc["pass"]:
            raise ValueError(f"QC-failing cell included: {cell.ids['cell']}")
        if {key: cell.ids[key] for key in replica_keys} != reference_ids:
            raise ValueError("All cells must belong to the same replica")
        if not np.array_equal(np.asarray(cell.parameters["edges_m"]), edges):
            raise ValueError("All cells in one replica must use identical bins")
        if "intra_contribution_density_um_inv" not in cell.distributions:
            cell.distributions = add_distribution_densities(cell.distributions, edges)
        if "positive_intra" not in cell.summaries:
            cell.summaries["positive_intra"] = summarize_positive_intra(
                cell.distributions["intra_contribution"], edges
            )

    total_stack = np.stack([cell.distributions["total"] for cell in cells])
    inter_stack = np.stack(
        [cell.distributions["inter_contribution"] for cell in cells]
    )
    intra_stack = np.stack(
        [cell.distributions["intra_contribution"] for cell in cells]
    )
    background_stack = np.stack(
        [cell.distributions["inter_conditional"] for cell in cells]
    )
    beta_values = np.asarray([cell.distributions["beta"] for cell in cells])
    total = total_stack.mean(axis=0)
    inter = inter_stack.mean(axis=0)
    intra = intra_stack.mean(axis=0)
    beta = float(beta_values.mean())
    distributions = add_distribution_densities(
        {
            "total": total,
            "inter_contribution": inter,
            "intra_contribution": intra,
            "inter_conditional": _conditional_shape(inter, beta),
            "intra_conditional": _conditional_shape(intra, 1.0 - beta),
            "average_inter_conditional_shape": background_stack.mean(axis=0),
            "beta": beta,
            "aggregation": "equal_cell_mean",
        },
        edges,
    )
    positive_intra = summarize_positive_intra(intra, edges)

    per_cell_counts = [
        {
            "cell_id": cell.ids["cell"],
            "localizations": cell.counts["localizations"],
            "frames": cell.counts["frames"],
            "total_candidate_pairs": cell.counts["total_candidate_pairs"],
            "total_retained_pairs": cell.counts["total_retained_pairs"],
            "inter_candidate_pairs": cell.counts["inter_candidate_pairs"],
            "inter_retained_pairs": cell.counts["inter_retained_pairs"],
        }
        for cell in cells
    ]
    total_candidate = int(
        sum(cell.counts["total_candidate_pairs"] for cell in cells)
    )
    total_retained = int(
        sum(cell.counts["total_retained_pairs"] for cell in cells)
    )
    inter_candidate = int(
        sum(cell.counts["inter_candidate_pairs"] for cell in cells)
    )
    inter_retained = int(
        sum(cell.counts["inter_retained_pairs"] for cell in cells)
    )
    reconstruction_error = float(np.max(np.abs(total - inter - intra)))
    excluded_cells = list(excluded_cells or [])
    flags: list[str] = []
    if excluded_cells:
        flags.append("excluded_cells")
    if np.any(intra < 0):
        flags.append("negative_intra_bins")
    if reconstruction_error > 1e-12:
        flags.append("reconstruction_error")

    return ReplicaResult(
        ids=reference_ids,
        source=dict(source or first.source),
        cells={
            "discovered_cell_ids": list(
                discovered_cell_ids
                if discovered_cell_ids is not None
                else [cell.ids["cell"] for cell in cells]
            ),
            "included_cell_ids": [cell.ids["cell"] for cell in cells],
            "excluded_cells": excluded_cells,
            "discovered_count": len(
                discovered_cell_ids if discovered_cell_ids is not None else cells
            ),
            "included_count": len(cells),
            "excluded_count": len(excluded_cells),
        },
        parameters=dict(first.parameters),
        counts={
            "n_cells": len(cells),
            "total_pair_count_per_tau": {str(config.tau_frames): total_retained},
            "total_candidate_pairs": total_candidate,
            "total_retained_pairs": total_retained,
            "inter_candidate_pairs": inter_candidate,
            "inter_retained_pairs": inter_retained,
            "per_cell": per_cell_counts,
        },
        distributions=distributions,
        summaries={"positive_intra": positive_intra},
        qc={
            "pass": reconstruction_error <= 1e-12,
            "status": "pass" if not flags else "pass_with_warnings",
            "flags": flags,
            "metrics": {
                "reconstruction_error": reconstruction_error,
                "normalization_error": float(abs(total.sum() - 1)),
                "beta_identity_error": float(abs(inter.sum() - beta)),
                "negative_intra_bins": int(np.count_nonzero(intra < 0)),
                "negative_intra_mass": float(-np.minimum(intra, 0).sum()),
            },
        },
        provenance=deepcopy(
            provenance if provenance is not None else getattr(first, "provenance", {})
        ),
        schema_version="tau1-replica-cell-balanced-v4",
    )

def aggregate_genotype(
    cell_results: Iterable[CellResult], config: Tau1Config | None = None
) -> GenotypeResult:
    """Legacy version-1 equal-cell aggregation."""
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
    reconstruction_error = float(np.max(np.abs(total - inter - intra)))
    flags = ["negative_intra_bins"] if np.any(intra < 0) else []
    if len(cells) == 1:
        flags.append("single_cell_no_biological_interval")
    first = cells[0]
    return GenotypeResult(
        ids={
            "condition": first.ids["condition"],
            "target": first.ids["target"],
            "genotype": first.ids.get("genotype", "UNASSIGNED"),
        },
        cells={
            "included_cell_ids": [cell.ids["cell"] for cell in cells],
            "replicate_ids": [cell.ids["replicate"] for cell in cells],
            "included_count": len(cells),
            "excluded_count": 0,
        },
        parameters=dict(first.parameters),
        distributions=add_distribution_densities(
            {
                "total": total,
                "inter_contribution": inter,
                "intra_contribution": intra,
                "inter_conditional": _conditional_shape(inter, beta),
                "intra_conditional": _conditional_shape(intra, 1 - beta),
                "beta": beta,
            },
            np.asarray(reference_edges),
        ),
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


def _conditional_shape(contribution: np.ndarray, mass: float) -> np.ndarray:
    if abs(mass) <= 100 * np.finfo(float).eps:
        return np.full_like(contribution, np.nan)
    return contribution / mass


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
            selected.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
        boot_total[draw] = np.mean(
            [cells[i].distributions["total"] for i in selected], axis=0
        )
        boot_inter[draw] = np.mean(
            [cells[i].distributions["inter_contribution"] for i in selected], axis=0
        )
        boot_intra[draw] = np.mean(
            [cells[i].distributions["intra_contribution"] for i in selected], axis=0
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


def _required_positive_float(value: Any, name: str) -> float:
    number = _optional_float(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be a positive finite value")
    return number
