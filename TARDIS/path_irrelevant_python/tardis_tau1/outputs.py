from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import Tau1Config
from .core import CellResult, GenotypeResult, ReplicaResult

_DENSITY_KEYS = (
    "total_density_um_inv",
    "inter_conditional_density_um_inv",
    "inter_contribution_density_um_inv",
    "intra_contribution_density_um_inv",
    "intra_conditional_density_um_inv",
    "average_inter_conditional_density_um_inv",
    "average_intra_conditional_density_um_inv",
)


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
    arrays: dict[str, np.ndarray] = {
        "edges_m": np.asarray(result.parameters["edges_m"]),
        "bin_mids_m": np.asarray(result.parameters["bin_mids_m"]),
        "total": np.asarray(distributions["total"]),
        "inter_contribution": np.asarray(distributions["inter_contribution"]),
        "intra_contribution": np.asarray(distributions["intra_contribution"]),
        "inter_conditional": np.asarray(distributions["inter_conditional"]),
        "intra_conditional": np.asarray(distributions["intra_conditional"]),
        "beta": np.asarray(distributions["beta"]),
        "schema_version": np.asarray(result.schema_version),
    }
    _add_density_arrays(arrays, distributions)
    _add_summary_arrays(arrays, getattr(result, "summaries", {}))
    _add_provenance_arrays(arrays, getattr(result, "provenance", {}))
    if isinstance(result, CellResult):
        arrays.update(
            total_count=np.asarray(result.counts["total_histogram"]),
            inter_background_count=np.asarray(result.counts["inter_histogram"]),
            beta_raw=np.asarray(result.distributions["beta_raw"]),
        )
    np.savez_compressed(output_dir / f"{level}_result.npz", **arrays)
    metadata = _json_safe(asdict(result))
    (output_dir / f"{level}_result.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_distribution_csv(result, output_dir / "distributions.csv")
    if config.save_plots:
        _write_diagnostic_plots(result, output_dir, level)
        _write_intra_area_normalized_plots(result, output_dir, level)


def save_replica_result(
    result: ReplicaResult,
    cell_results: list[CellResult],
    output_dir: str | Path,
    config: Tau1Config,
) -> None:
    """Save the canonical equal-cell replica and all contributing cells."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    distributions = result.distributions
    cell_intra = np.stack(
        [cell.distributions["intra_contribution"] for cell in cell_results]
    )
    cell_intra_density = np.stack(
        [cell.distributions["intra_contribution_density_um_inv"] for cell in cell_results]
    )
    cell_intra_conditional = np.stack(
        [cell.distributions["intra_conditional"] for cell in cell_results]
    )
    cell_intra_conditional_density = np.stack(
        [cell.distributions["intra_conditional_density_um_inv"] for cell in cell_results]
    )
    arrays: dict[str, np.ndarray] = {
        "edges_m": np.asarray(result.parameters["edges_m"]),
        "bin_mids_m": np.asarray(result.parameters["bin_mids_m"]),
        "total": np.asarray(distributions["total"]),
        "inter_contribution": np.asarray(distributions["inter_contribution"]),
        "intra_contribution": np.asarray(distributions["intra_contribution"]),
        "inter_conditional": np.asarray(distributions["inter_conditional"]),
        "intra_conditional": np.asarray(distributions["intra_conditional"]),
        "average_inter_conditional_shape": np.asarray(
            distributions["average_inter_conditional_shape"]
        ),
        "average_intra_conditional_shape": np.asarray(
            distributions["average_intra_conditional_shape"]
        ),
        "beta": np.asarray(distributions["beta"]),
        "cell_ids": np.asarray([cell.ids["cell"] for cell in cell_results]),
        "cell_intra_contributions": cell_intra,
        "cell_intra_density_um_inv": cell_intra_density,
        "cell_intra_conditional_shapes": cell_intra_conditional,
        "cell_intra_conditional_density_um_inv": cell_intra_conditional_density,
        "schema_version": np.asarray(result.schema_version),
    }
    _add_density_arrays(arrays, distributions)
    _add_summary_arrays(arrays, result.summaries)
    _add_provenance_arrays(arrays, result.provenance)
    np.savez_compressed(output_dir / "replica_result.npz", **arrays)
    (output_dir / "replica_result.json").write_text(
        json.dumps(_json_safe(asdict(result)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_replica_distribution_csv(result, output_dir / "replica_distributions.csv")
    _write_cell_summary(result, cell_results, output_dir / "cell_summary.csv")
    _write_replica_summary(result, output_dir / "replica_summary.csv")
    _write_peak_normalized_shape_summary(
        result, cell_results, output_dir / "cell_peak_normalized_shape_summary.csv"
    )
    cells_dir = output_dir / "cells"
    for cell in cell_results:
        save_result(cell, cells_dir / safe_component(cell.ids["cell"]), config, "cell")
    obsolete = (
        "cell_balanced_distributions.csv",
        "pair_weighted_distributions.csv",
        "mode_comparison.json",
        "aggregation_comparison.png",
    )
    for name in obsolete:
        path = output_dir / name
        if path.is_file():
            path.unlink()
    if config.save_plots:
        _write_diagnostic_plots(result, output_dir, "replica", stem="replica_distribution")
        _write_intra_area_normalized_plots(result, output_dir, "replica")
        _write_all_cell_intra_plots(result, cell_results, output_dir)
        _write_peak_normalized_all_cell_intra_plots(result, cell_results, output_dir)


def safe_component(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))
    return cleaned or "UNASSIGNED"


def _write_distribution_csv(
    result: CellResult | GenotypeResult, path: Path
) -> None:
    d = result.distributions
    edges = np.asarray(result.parameters["edges_m"])
    mids = np.asarray(result.parameters["bin_mids_m"])
    n_bins = mids.size
    if isinstance(result, CellResult):
        total_count = np.asarray(result.counts["total_histogram"])
        inter_count = np.asarray(result.counts["inter_histogram"])
        total_pairs = float(result.counts["total_retained_pairs"])
        inter_expected = np.asarray(d["inter_contribution"]) * total_pairs
        intra_expected = np.asarray(d["intra_contribution"]) * total_pairs
    else:
        total_count = [""] * n_bins
        inter_count = [""] * n_bins
        inter_expected = [""] * n_bins
        intra_expected = [""] * n_bins
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "distance_left_m",
                "distance_right_m",
                "distance_mid_m",
                "total_count",
                "inter_background_count",
                "total_probability",
                "inter_background_probability",
                "inter_contribution_probability",
                "intra_contribution_probability",
                "inter_expected_count",
                "intra_expected_count",
                "total_density_um_inv",
                "inter_background_density_um_inv",
                "inter_contribution_density_um_inv",
                "intra_contribution_density_um_inv",
                "intra_conditional_density_um_inv",
            ]
        )
        writer.writerows(
            zip(
                edges[:-1],
                edges[1:],
                mids,
                total_count,
                inter_count,
                d["total"],
                d["inter_conditional"],
                d["inter_contribution"],
                d["intra_contribution"],
                inter_expected,
                intra_expected,
                d["total_density_um_inv"],
                d["inter_conditional_density_um_inv"],
                d["inter_contribution_density_um_inv"],
                d["intra_contribution_density_um_inv"],
                d["intra_conditional_density_um_inv"],
            )
        )


def _write_replica_distribution_csv(result: ReplicaResult, path: Path) -> None:
    d = result.distributions
    edges = np.asarray(result.parameters["edges_m"])
    mids = np.asarray(result.parameters["bin_mids_m"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "aggregation",
                "distance_left_m",
                "distance_right_m",
                "distance_mid_m",
                "total_probability",
                "inter_background_probability",
                "inter_contribution_probability",
                "intra_contribution_probability",
                "total_density_um_inv",
                "inter_background_density_um_inv",
                "inter_contribution_density_um_inv",
                "intra_contribution_density_um_inv",
                "average_intra_conditional_shape_probability",
                "average_intra_conditional_shape_density_um_inv",
            ]
        )
        writer.writerows(
            zip(
                [d["aggregation"]] * mids.size,
                edges[:-1],
                edges[1:],
                mids,
                d["total"],
                d["inter_conditional"],
                d["inter_contribution"],
                d["intra_contribution"],
                d["total_density_um_inv"],
                d["inter_conditional_density_um_inv"],
                d["inter_contribution_density_um_inv"],
                d["intra_contribution_density_um_inv"],
                d["average_intra_conditional_shape"],
                d["average_intra_conditional_density_um_inv"],
            )
        )


def _write_cell_summary(
    result: ReplicaResult, cell_results: list[CellResult], path: Path
) -> None:
    pair_counts = {row["cell_id"]: row for row in result.counts["per_cell"]}
    positive_fields = list(result.summaries["positive_intra"])
    fields = [
        "cell_id",
        "localizations",
        "frames",
        "beta_raw",
        "beta",
        "beta_was_clipped",
        "total_candidate_pairs",
        "total_retained_pairs",
        "inter_candidate_pairs",
        "inter_retained_pairs",
        "negative_intra_bins",
        "negative_intra_mass",
        *positive_fields,
        "qc_status",
        "qc_flags",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cell_results:
            counts = pair_counts[cell.ids["cell"]]
            row = {
                "cell_id": cell.ids["cell"],
                "localizations": cell.counts["localizations"],
                "frames": cell.counts["frames"],
                "beta_raw": cell.distributions["beta_raw"],
                "beta": cell.distributions["beta"],
                "beta_was_clipped": cell.distributions["beta_was_clipped"],
                "total_candidate_pairs": counts["total_candidate_pairs"],
                "total_retained_pairs": counts["total_retained_pairs"],
                "inter_candidate_pairs": counts["inter_candidate_pairs"],
                "inter_retained_pairs": counts["inter_retained_pairs"],
                "negative_intra_bins": cell.qc["metrics"]["negative_intra_bins"],
                "negative_intra_mass": cell.qc["metrics"]["negative_intra_mass"],
                "qc_status": cell.qc["status"],
                "qc_flags": ";".join(cell.qc["flags"]),
            }
            row.update(_csv_safe_mapping(cell.summaries["positive_intra"]))
            writer.writerow(row)


def _write_replica_summary(result: ReplicaResult, path: Path) -> None:
    provenance = result.provenance
    row = {
        "condition": result.ids["condition"],
        "target": result.ids["target"],
        "replica": result.ids["replicate"],
        "aggregation": result.distributions["aggregation"],
        "n_cells": result.counts["n_cells"],
        "beta": result.distributions["beta"],
        "negative_intra_bins": result.qc["metrics"]["negative_intra_bins"],
        "negative_intra_mass": result.qc["metrics"]["negative_intra_mass"],
        **_csv_safe_mapping(result.summaries["positive_intra"]),
        "schema_version": result.schema_version,
        "input_sha256": provenance.get("input", {}).get("sha256", ""),
        "input_file_bytes": provenance.get("input", {}).get("file_bytes", ""),
        "analysis_runtime_seconds": provenance.get("execution", {}).get(
            "analysis_runtime_seconds", ""
        ),
        "git_commit": provenance.get("git", {}).get("commit") or "",
        "git_dirty": provenance.get("git", {}).get("dirty", ""),
        "package_version": provenance.get("package", {}).get("version") or "",
        "python_version": provenance.get("python", {}).get("version", ""),
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def _split_paths(output_dir: Path, stem: str) -> tuple[Path, Path]:
    """Return the standard full-range and zoom file names for one plot family."""
    return (
        output_dir / f"{stem}_full_range.png",
        output_dir / f"{stem}_zoom_0p5um.png",
    )


def _finish_single_axis(
    fig: Any, axis: Any, path: Path, *, xlim: tuple[float, float], title: str,
    ylabel: str, suptitle: str, legend: bool = False,
) -> None:
    axis.set_xlim(*xlim)
    axis.set_title(title)
    axis.set_xlabel("Distance (μm)")
    axis.set_ylabel(ylabel)
    if legend:
        axis.legend(frameon=False, loc="upper right")
    fig.suptitle(suptitle)
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    _close_figure(fig)


def _write_diagnostic_plots(
    result: CellResult | GenotypeResult | ReplicaResult, output_dir: Path,
    level: str, *, stem: str = "diagnostic",
) -> None:
    """Write separate full-range and Intra-only zoom diagnostic figures."""
    x_um = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    edges_um = np.asarray(result.parameters["edges_m"]) * 1e6
    title = f"{level.capitalize()} τ=1 distribution · β={result.distributions['beta']:.4f}"
    full_path, zoom_path = _split_paths(output_dir, stem)
    for path, zoom in ((full_path, False), (zoom_path, True)):
        plt = _pyplot()
        fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        if zoom:
            _plot_intra_distribution(axis, x_um, result.distributions)
        else:
            _plot_distributions(axis, x_um, result.distributions)
        _finish_single_axis(
            fig, axis, path,
            xlim=(0.0, min(0.5, float(edges_um[-1]))) if zoom else (edges_um[0], edges_um[-1]),
            title="Zoom: 0–0.5 μm · Intra only" if zoom else "Full range",
            ylabel="Probability density (μm⁻¹)", suptitle=title, legend=zoom,
        )


def _write_intra_area_normalized_plots(
    result: CellResult | ReplicaResult, output_dir: Path, level: str
) -> None:
    """Write signed Intra densities normalized by their net mass (1 - beta)."""
    x_um = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    edges_um = np.asarray(result.parameters["edges_m"]) * 1e6
    if isinstance(result, ReplicaResult):
        density = np.asarray(
            result.distributions["average_intra_conditional_density_um_inv"]
        )
        summary = result.summaries.get("intra_shape", {})
    else:
        density = np.asarray(result.distributions["intra_conditional_density_um_inv"])
        summary = result.summaries.get("positive_intra", {})
    subtitle = ""
    if summary:
        subtitle = (
            f" · peak {summary['peak_distance_um'] * 1e3:.0f} nm"
            f" · mean {summary['positive_weighted_mean_distance_um'] * 1e3:.0f} nm"
            f" · median {summary['positive_weighted_median_distance_um'] * 1e3:.0f} nm"
        )
    full_path, zoom_path = _split_paths(output_dir, "intra_area_normalized")
    for path, zoom in ((full_path, False), (zoom_path, True)):
        plt = _pyplot()
        fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        axis.plot(x_um, density, color="#1baf7a", linewidth=3.2, label="Intra")
        _style_axis(axis)
        _finish_single_axis(
            fig, axis, path,
            xlim=(0.0, min(0.5, float(edges_um[-1]))) if zoom else (edges_um[0], edges_um[-1]),
            title="Zoom: 0–0.5 μm" if zoom else "Full range",
            ylabel="Signed-net-mass-normalized Intra density (μm⁻¹)",
            suptitle=f"{level.capitalize()} Intra · signed net mass normalized{subtitle}", legend=zoom,
        )


def _build_distribution_figure(result: Any, title: str) -> tuple[Any, Any]:
    plt = _pyplot()
    x_um = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    edges_um = np.asarray(result.parameters["edges_m"]) * 1e6
    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.8), sharey=True, constrained_layout=True
    )
    _plot_distributions(axes[0], x_um, result.distributions)
    _plot_intra_distribution(axes[1], x_um, result.distributions)
    for ax in axes:
        ax.set_xlabel("Distance (μm)")
    axes[0].set_xlim(edges_um[0], edges_um[-1])
    axes[0].set_title("Full range")
    zoom_max = min(0.5, float(edges_um[-1]))
    axes[1].set_xlim(0.0, zoom_max)
    axes[1].set_title(f"Zoom: 0–{zoom_max:g} μm · Intra only")
    axes[0].set_ylabel("Probability density (μm⁻¹)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(title, y=1.02)
    return fig, axes


def _write_all_cell_intra_plots(
    result: ReplicaResult, cell_results: list[CellResult], output_dir: Path
) -> None:
    """Write separate full-range and zoom overlays for all cells in a replica."""
    plt = _pyplot()
    x_um = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    edges_um = np.asarray(result.parameters["edges_m"]) * 1e6
    colors = plt.get_cmap("tab10").colors
    linestyles = ("-", "--", "-.", ":")
    title = (
        f"{result.ids['condition']} · {result.ids['target']} · "
        f"{result.ids['replicate']} · tau=1 · n={len(cell_results)} cells"
    )
    full_path, zoom_path = _split_paths(output_dir, "all_cell_intra")
    for path, zoom in ((full_path, False), (zoom_path, True)):
        fig, axis = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
        for index, cell in enumerate(cell_results):
            axis.plot(
                x_um, cell.distributions["intra_contribution_density_um_inv"],
                color=colors[index % len(colors)],
                linestyle=linestyles[(index // len(colors)) % len(linestyles)],
                linewidth=2.0, alpha=0.85, label=cell.ids["cell"],
            )
        axis.plot(x_um, result.distributions["intra_contribution_density_um_inv"], color="#111111", linewidth=3.0, label="Replica mean Intra", zorder=10)
        _style_axis(axis)
        _finish_single_axis(
            fig, axis, path,
            xlim=(0.0, min(0.5, float(edges_um[-1]))) if zoom else (edges_um[0], edges_um[-1]),
            title="Zoom: 0–0.5 μm" if zoom else "Full range",
            ylabel="Intra contribution density (μm⁻¹)", suptitle=title, legend=zoom,
        )


def _build_all_cell_intra_figure(
    result: ReplicaResult, cell_results: list[CellResult]
) -> tuple[Any, Any]:
    plt = _pyplot()
    x_um = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    edges_um = np.asarray(result.parameters["edges_m"]) * 1e6
    fig, axes = plt.subplots(
        1, 2, figsize=(14.0, 5.8), sharey=True, constrained_layout=True
    )
    colors = plt.get_cmap("tab10").colors
    linestyles = ("-", "--", "-.", ":")
    for ax in axes:
        for index, cell in enumerate(cell_results):
            ax.plot(
                x_um,
                cell.distributions["intra_contribution_density_um_inv"],
                color=colors[index % len(colors)],
                linestyle=linestyles[(index // len(colors)) % len(linestyles)],
                linewidth=2.0,
                alpha=0.85,
                label=cell.ids["cell"],
            )
        ax.plot(
            x_um,
            result.distributions["intra_contribution_density_um_inv"],
            color="#111111",
            linewidth=3.0,
            label="Replica mean Intra",
            zorder=10,
        )
        _style_axis(ax)
        ax.set_xlabel("Distance (μm)")
    axes[0].set_xlim(edges_um[0], edges_um[-1])
    axes[0].set_title("Full range")
    zoom_max = min(0.5, float(edges_um[-1]))
    axes[1].set_xlim(0.0, zoom_max)
    axes[1].set_title(f"Zoom: 0–{zoom_max:g} μm")
    axes[0].set_ylabel("Intra contribution density (μm⁻¹)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        fontsize=8,
    )
    fig.suptitle(
        f"{result.ids['condition']} · {result.ids['target']} · "
        f"{result.ids['replicate']} · tau=1 · n={len(cell_results)} cells"
    )
    return fig, axes


def _write_peak_normalized_all_cell_intra_plots(
    result: ReplicaResult, cell_results: list[CellResult], output_dir: Path
) -> None:
    """Write separate full-range and zoom overlays after per-cell peak scaling."""
    plt = _pyplot()
    x_um = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    edges_um = np.asarray(result.parameters["edges_m"]) * 1e6
    normalized, _ = _peak_normalized_intra_curves(cell_results)
    mean_curve = np.nanmean(normalized, axis=0)
    colors = plt.get_cmap("tab10").colors
    linestyles = ("-", "--", "-.", ":")
    title = (
        f"{result.ids['condition']} · {result.ids['target']} · "
        f"{result.ids['replicate']} · tau=1 · peak-normalized Intra"
    )
    full_path, zoom_path = _split_paths(output_dir, "all_cell_intra_peak_normalized")
    for path, zoom in ((full_path, False), (zoom_path, True)):
        fig, axis = plt.subplots(figsize=(8.5, 5.8), constrained_layout=True)
        for index, (cell, curve) in enumerate(zip(cell_results, normalized)):
            if np.isfinite(curve).any():
                axis.plot(x_um, curve, color=colors[index % len(colors)], linestyle=linestyles[(index // len(colors)) % len(linestyles)], linewidth=2.0, alpha=0.85, label=cell.ids["cell"])
        axis.plot(x_um, mean_curve, color="#111111", linewidth=3.0, label="Mean peak-normalized Intra", zorder=10)
        _style_axis(axis)
        _finish_single_axis(
            fig, axis, path,
            xlim=(0.0, min(0.5, float(edges_um[-1]))) if zoom else (edges_um[0], edges_um[-1]),
            title="Zoom: 0–0.5 μm" if zoom else "Full range",
            ylabel="Intra density / cell positive peak", suptitle=title, legend=zoom,
        )


def _build_peak_normalized_all_cell_intra_figure(
    result: ReplicaResult, cell_results: list[CellResult]
) -> tuple[Any, Any]:
    """Plot signed Intra curves after scaling each cell's positive peak to one."""
    plt = _pyplot()
    x_um = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    edges_um = np.asarray(result.parameters["edges_m"]) * 1e6
    normalized, _ = _peak_normalized_intra_curves(cell_results)
    mean_curve = np.nanmean(normalized, axis=0)
    fig, axes = plt.subplots(
        1, 2, figsize=(14.0, 5.8), sharey=True, constrained_layout=True
    )
    colors = plt.get_cmap("tab10").colors
    linestyles = ("-", "--", "-.", ":")
    for ax in axes:
        for index, (cell, curve) in enumerate(zip(cell_results, normalized)):
            if not np.isfinite(curve).any():
                continue
            ax.plot(
                x_um,
                curve,
                color=colors[index % len(colors)],
                linestyle=linestyles[(index // len(colors)) % len(linestyles)],
                linewidth=2.0,
                alpha=0.85,
                label=cell.ids["cell"],
            )
        ax.plot(
            x_um,
            mean_curve,
            color="#111111",
            linewidth=3.0,
            label="Mean peak-normalized Intra",
            zorder=10,
        )
        _style_axis(ax)
        ax.set_xlabel("Distance (μm)")
    axes[0].set_xlim(edges_um[0], edges_um[-1])
    axes[0].set_title("Full range")
    zoom_max = min(0.5, float(edges_um[-1]))
    axes[1].set_xlim(0.0, zoom_max)
    axes[1].set_title(f"Zoom: 0–{zoom_max:g} μm")
    axes[0].set_ylabel("Intra density / cell positive peak")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        bbox_to_anchor=(1.01, 0.5),
        loc="center left",
        fontsize=8,
    )
    fig.suptitle(
        f"{result.ids['condition']} · {result.ids['target']} · "
        f"{result.ids['replicate']} · tau=1 · peak-normalized Intra"
    )
    return fig, axes


def _peak_normalized_intra_curves(
    cell_results: list[CellResult],
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed curves normalized by their positive maxima and those maxima."""
    raw = np.asarray(
        [
            cell.distributions["intra_conditional_density_um_inv"]
            for cell in cell_results
        ],
        dtype=float,
    )
    peaks = np.max(raw, axis=1)
    normalized = np.full_like(raw, np.nan)
    valid = peaks > 0
    normalized[valid] = raw[valid] / peaks[valid, np.newaxis]
    return normalized, peaks


def _write_peak_normalized_shape_summary(
    result: ReplicaResult, cell_results: list[CellResult], path: Path
) -> None:
    """Write reproducible shape-comparison metrics for peak-normalized curves."""
    x_um = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    normalized, peaks = _peak_normalized_intra_curves(cell_results)
    zoom_mask = x_um <= min(0.5, float(x_um[-1]))
    fields = [
        "cell_id",
        "positive_peak_density_um_inv",
        "peak_distance_um",
        "shape_window_um",
        "pearson_correlation_to_leave_one_out_mean_peak_normalized_curve",
        "rmse_to_leave_one_out_mean_peak_normalized_curve",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (cell, curve, peak) in enumerate(
            zip(cell_results, normalized, peaks)
        ):
            window = curve[zoom_mask]
            other_curves = np.delete(normalized, index, axis=0)
            reference = np.nanmean(other_curves, axis=0)[zoom_mask]
            finite = np.isfinite(window) & np.isfinite(reference)
            correlation = float("nan")
            rmse = float("nan")
            if finite.any():
                rmse = float(np.sqrt(np.mean((window[finite] - reference[finite]) ** 2)))
            if (
                finite.sum() >= 2
                and np.std(window[finite]) > 0
                and np.std(reference[finite]) > 0
            ):
                correlation = float(np.corrcoef(window[finite], reference[finite])[0, 1])
            peak_index = int(np.argmax(curve)) if np.isfinite(curve).any() else -1
            writer.writerow(
                {
                    "cell_id": cell.ids["cell"],
                    "positive_peak_density_um_inv": peak,
                    "peak_distance_um": "" if peak_index < 0 else x_um[peak_index],
                    "shape_window_um": 0.5,
                    "pearson_correlation_to_leave_one_out_mean_peak_normalized_curve": correlation,
                    "rmse_to_leave_one_out_mean_peak_normalized_curve": rmse,
                }
            )


def _plot_distributions(ax: Any, x_um: np.ndarray, d: dict[str, Any]) -> None:
    ax.plot(
        x_um,
        d["total_density_um_inv"],
        color="#2a78d6",
        linewidth=3.0,
        label="Total",
    )
    ax.plot(
        x_um,
        d["inter_contribution_density_um_inv"],
        color="#eb6834",
        linewidth=2.0,
        linestyle="--",
        label="Inter contribution",
    )
    ax.plot(
        x_um,
        d["intra_contribution_density_um_inv"],
        color="#1baf7a",
        linewidth=2.0,
        linestyle="-.",
        label="Intra contribution",
    )
    _style_axis(ax)


def _plot_intra_distribution(ax: Any, x_um: np.ndarray, d: dict[str, Any]) -> None:
    """Plot only signed Intra in zoom panels used for cell and replica QC."""
    ax.plot(
        x_um,
        d["intra_contribution_density_um_inv"],
        color="#1baf7a",
        linewidth=2.0,
        linestyle="-.",
        label="Intra contribution",
    )
    _style_axis(ax)


def _style_axis(ax: Any) -> None:
    ax.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
    ax.grid(axis="y", color="#dddddd", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _close_figure(fig: Any) -> None:
    _pyplot().close(fig)


def _add_density_arrays(
    arrays: dict[str, np.ndarray], distributions: dict[str, Any]
) -> None:
    for key in _DENSITY_KEYS:
        if key in distributions:
            arrays[key] = np.asarray(distributions[key])


def _add_summary_arrays(
    arrays: dict[str, np.ndarray], summaries: dict[str, Any]
) -> None:
    for key, value in summaries.get("positive_intra", {}).items():
        if isinstance(value, (int, float, np.number)):
            arrays[f"positive_intra_{key}"] = np.asarray(value)


def _add_provenance_arrays(
    arrays: dict[str, np.ndarray], provenance: dict[str, Any]
) -> None:
    values = {
        "input_sha256": provenance.get("input", {}).get("sha256"),
        "input_file_bytes": provenance.get("input", {}).get("file_bytes"),
        "package_version": provenance.get("package", {}).get("version"),
        "python_version": provenance.get("python", {}).get("version"),
        "git_commit": provenance.get("git", {}).get("commit"),
        "git_dirty": provenance.get("git", {}).get("dirty"),
        "analysis_runtime_seconds": provenance.get("execution", {}).get(
            "analysis_runtime_seconds"
        ),
    }
    for key, value in values.items():
        if value is not None:
            arrays[key] = np.asarray(value)


def _csv_safe_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: "" if isinstance(value, float) and not np.isfinite(value) else value
        for key, value in values.items()
    }


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


def load_replica_result(output_dir: str | Path) -> ReplicaResult:
    """Load and validate a saved v5 replica without loading its cells."""
    output = Path(output_dir)
    metadata = json.loads((output / "replica_result.json").read_text(encoding="utf-8"))
    with np.load(output / "replica_result.npz", allow_pickle=False) as saved:
        required = {
            "edges_m", "bin_mids_m", "total", "inter_contribution",
            "intra_contribution", "inter_conditional", "intra_conditional",
            "average_inter_conditional_shape", "average_intra_conditional_shape",
            "average_intra_conditional_density_um_inv",
            "cell_intra_conditional_shapes",
            "cell_intra_conditional_density_um_inv",
            "beta", "schema_version",
        }
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"Saved replica is missing arrays: {sorted(missing)}")
        schema = str(saved["schema_version"].item())
        if schema != metadata.get("schema_version"):
            raise ValueError("Replica JSON and NPZ schema versions do not match")
        if schema != "tau1-replica-cell-balanced-v5":
            raise ValueError(f"Unsupported replica schema: {schema}")
        edges = np.asarray(saved["edges_m"], dtype=float).copy()
        mids = np.asarray(saved["bin_mids_m"], dtype=float).copy()
        if (edges.ndim != 1 or mids.ndim != 1 or edges.size != mids.size + 1
                or not np.isfinite(edges).all() or not np.all(np.diff(edges) > 0)):
            raise ValueError("Saved replica has invalid distance bins")
        distributions = dict(metadata["distributions"])
        distribution_keys = (
            "total", "inter_contribution", "intra_contribution",
            "inter_conditional", "intra_conditional",
            "average_inter_conditional_shape", "average_intra_conditional_shape",
        )
        for key in distribution_keys:
            values = np.asarray(saved[key], dtype=float).copy()
            if (
                values.ndim != 1
                or values.size != mids.size
                or not np.isfinite(values).all()
            ):
                raise ValueError(f"Saved replica array {key} does not match bins")
            distributions[key] = values
        distributions["beta"] = float(np.asarray(saved["beta"]).item())
        for key in saved.files:
            if key.endswith("_density_um_inv") and np.asarray(saved[key]).ndim == 1:
                values = np.asarray(saved[key], dtype=float).copy()
                if values.size == mids.size:
                    distributions[key] = values
        cell_shapes = np.asarray(saved["cell_intra_conditional_shapes"], dtype=float)
        cell_shape_density = np.asarray(
            saved["cell_intra_conditional_density_um_inv"], dtype=float
        )
        expected_matrix_shape = (int(metadata["counts"]["n_cells"]), mids.size)
        if (
            cell_shapes.shape != expected_matrix_shape
            or cell_shape_density.shape != expected_matrix_shape
            or not np.isfinite(cell_shapes).all()
            or not np.isfinite(cell_shape_density).all()
        ):
            raise ValueError("Saved replica has invalid per-cell normalized Intra shapes")
        widths_um = np.diff(edges) * 1e6
        if not np.allclose(
            cell_shape_density * widths_um,
            cell_shapes,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Saved per-cell Intra shape probabilities and densities disagree")
        cell_shape_masses = cell_shapes.sum(axis=1)
        if not np.allclose(
            cell_shape_masses, 1.0, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Saved per-cell normalized Intra shapes must have signed mass 1")
        shape = distributions["average_intra_conditional_shape"]
        if not np.allclose(
            cell_shapes.mean(axis=0), shape, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Saved replica Intra shape is not the equal-cell mean")
        if not np.isclose(shape.sum(), 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError("Saved replica normalized Intra shape must have signed mass 1")
        shape_density = distributions["average_intra_conditional_density_um_inv"]
        if not np.allclose(
            shape_density * widths_um, shape, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Saved replica Intra shape probabilities and densities disagree")
        if not np.allclose(
            distributions["total"],
            distributions["inter_contribution"] + distributions["intra_contribution"],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Saved replica contribution decomposition is inconsistent")
    parameters = dict(metadata["parameters"])
    parameters["edges_m"] = edges
    parameters["bin_mids_m"] = mids
    return ReplicaResult(
        ids=dict(metadata["ids"]), source=dict(metadata["source"]),
        cells=dict(metadata["cells"]), parameters=parameters,
        counts=dict(metadata["counts"]), distributions=distributions,
        qc=dict(metadata["qc"]), schema_version=schema,
        summaries=dict(metadata.get("summaries", {})),
        provenance=dict(metadata.get("provenance", {})),
    )
