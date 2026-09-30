"""Cross-condition Intra comparisons for one molecular target."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from .core import summarize_positive_intra


def save_target_condition_comparison(
    target: str, condition_directories: dict[str, Path], output_dir: Path
) -> None:
    """Save reproducible cross-condition Intra curves and scalar summaries."""
    ordered = _ordered(condition_directories, ("WT", "TDK", "MDK"))
    records = {
        condition: _read_group_directory(path / "group")
        for condition, path in ordered.items()
        if (path / "group" / "group_mean_distributions.csv").is_file()
    }
    if not records:
        return
    _validate_distribution_grids(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_distribution_table(records, output_dir / "intra_contribution_by_distance.csv")
    _write_summary_table(records, output_dir / "intra_shape_summary.csv")
    _write_split_plots(target, records, output_dir, "intra_contribution", conditional=False)
    conditional = {condition: _conditional_record(path / "group") for condition, path in ordered.items() if condition in records}
    _validate_curve_grids(conditional)
    _write_conditional_table(conditional, output_dir / "intra_area_normalized_by_distance.csv")
    _write_shape_summary_table(
        conditional, output_dir / "intra_area_normalized_shape_summary.csv"
    )
    _write_split_plots(target, conditional, output_dir, "intra_area_normalized", conditional=True)
    peak_normalized = {condition: _peak_normalized_record(path / "group") for condition, path in ordered.items() if condition in records}
    _validate_curve_grids(peak_normalized)
    _write_conditional_table(peak_normalized, output_dir / "intra_peak_normalized_by_distance.csv")
    _write_peak_normalized_zoom_plot(target, peak_normalized, output_dir / "intra_peak_normalized_zoom_0p5um.png")


def save_condition_target_comparison(
    condition: str, target_directories: dict[str, Path], output_dir: Path
) -> None:
    """Compare all targets within one condition, in caller-supplied target order."""
    records = {
        target: _read_group_directory(path / "group")
        for target, path in target_directories.items()
        if (path / "group" / "group_mean_distributions.csv").is_file()
    }
    if not records:
        return
    _validate_distribution_grids(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_distribution_table(records, output_dir / "targets_intra_contribution_by_distance.csv")
    _write_summary_table(records, output_dir / "targets_intra_shape_summary.csv")
    _write_split_plots(condition, records, output_dir, "targets_intra_contribution", conditional=False)
    conditional_records = {target: _conditional_record(path / "group") for target, path in target_directories.items() if target in records}
    _validate_curve_grids(conditional_records)
    _write_conditional_table(conditional_records, output_dir / "targets_intra_area_normalized_by_distance.csv")
    _write_shape_summary_table(
        conditional_records,
        output_dir / "targets_intra_area_normalized_shape_summary.csv",
    )
    _write_split_plots(condition, conditional_records, output_dir, "targets_intra_area_normalized", conditional=True)
    peak_normalized_records = {target: _peak_normalized_record(path / "group") for target, path in target_directories.items() if target in records}
    _validate_curve_grids(peak_normalized_records)
    _write_conditional_table(peak_normalized_records, output_dir / "targets_intra_peak_normalized_by_distance.csv")
    _write_peak_normalized_zoom_plot(condition, peak_normalized_records, output_dir / "targets_intra_peak_normalized_zoom_0p5um.png")


def _ordered(values: dict[str, Path], preferred: tuple[str, ...]) -> dict[str, Path]:
    keys = [key for key in preferred if key in values] + sorted(set(values) - set(preferred))
    return {key: values[key] for key in keys}


def _read_group_directory(group_dir: Path) -> dict[str, Any]:
    distribution = np.genfromtxt(
        group_dir / "group_mean_distributions.csv",
        delimiter=",",
        names=True,
        dtype=float,
        encoding="utf-8",
    )
    summary: dict[str, dict[str, float]] = {}
    summary_path = group_dir / "replica_metric_summary.csv"
    if not summary_path.is_file():
        summary_path = group_dir / "group_summary.csv"
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            summary[row["metric"]] = {key: float(value) for key, value in row.items() if key != "metric"}
    mean_curve_summary: dict[str, float] = {}
    scoped_path = group_dir / "group_mean_curve_summary.csv"
    if scoped_path.is_file():
        with scoped_path.open(newline="", encoding="utf-8") as handle:
            mean_curve_summary = {
                row["metric"]: float(row["value"]) for row in csv.DictReader(handle)
            }
    if not mean_curve_summary:
        edges = np.r_[
            distribution["distance_left_m"][0],
            distribution["distance_right_m"],
        ]
        widths_um = np.diff(edges) * 1e6
        contribution = (
            distribution["intra_contribution_mean_density_um_inv"] * widths_um
        )
        mean_curve_summary = summarize_positive_intra(contribution, edges)
    return {
        "distribution": distribution,
        "summary": summary,
        "group_mean_curve_summary": mean_curve_summary,
    }


def _validate_distribution_grids(records: dict[str, dict[str, Any]]) -> None:
    """Require every comparison record to use the same finite distance bins."""
    reference_name, reference = next(iter(records.items()))
    reference_grid = _distribution_grid(reference["distribution"], reference_name)
    for name, record in list(records.items())[1:]:
        grid = _distribution_grid(record["distribution"], name)
        if any(
            not np.array_equal(candidate, expected)
            for candidate, expected in zip(grid, reference_grid)
        ):
            raise ValueError(
                f"Comparison distance bins do not match: {name} differs from "
                f"{reference_name}"
            )


def _distribution_grid(
    distribution: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields = ("distance_left_m", "distance_right_m", "distance_mid_m")
    if distribution.ndim != 1 or not distribution.size:
        raise ValueError(f"Comparison distribution for {name} must be non-empty")
    arrays = tuple(np.asarray(distribution[field], dtype=float) for field in fields)
    if not all(np.isfinite(values).all() for values in arrays):
        raise ValueError(f"Comparison distance bins for {name} must be finite")
    if not np.all(np.diff(arrays[2]) > 0):
        raise ValueError(f"Comparison distance bins for {name} must increase")
    return arrays


def _validate_curve_grids(records: dict[str, dict[str, Any]]) -> None:
    reference_name, reference = next(iter(records.items()))
    reference_x = np.asarray(reference["x_m"], dtype=float)
    for name, record in list(records.items())[1:]:
        if not np.array_equal(np.asarray(record["x_m"], dtype=float), reference_x):
            raise ValueError(
                f"Comparison distance bins do not match: {name} differs from "
                f"{reference_name}"
            )


def _load_replica_density_arrays(
    group_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(group_dir / "group_result.npz") as arrays:
        edges = np.asarray(arrays["edges_m"], dtype=float).copy()
        mids = np.asarray(arrays["bin_mids_m"], dtype=float).copy()
        try:
            raw = np.asarray(
                arrays[
                    "average_intra_conditional_shape_replica_density_um_inv"
                ],
                dtype=float,
            ).copy()
        except KeyError as exc:
            raise ValueError(
                f"Group output {group_dir} predates cell-normalized Intra shapes; "
                "recompute replicas and group outputs"
            ) from exc
    if edges.ndim != 1 or mids.ndim != 1 or raw.ndim != 2:
        raise ValueError(f"Invalid group distribution dimensions in {group_dir}")
    if edges.size != mids.size + 1 or raw.shape[1] != mids.size:
        raise ValueError(f"Group distribution bins do not align in {group_dir}")
    if not (
        np.isfinite(edges).all()
        and np.isfinite(mids).all()
        and np.isfinite(raw).all()
    ):
        raise ValueError(f"Group distributions must be finite in {group_dir}")
    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"Group distance edges must increase in {group_dir}")
    return edges, mids, raw


def _write_distribution_table(records: dict[str, dict[str, Any]], path: Path) -> None:
    first = next(iter(records.values()))["distribution"]
    fields = ["distance_left_m", "distance_right_m", "distance_mid_m"]
    for condition in records:
        fields.extend(
            [f"{condition}_intra_mean_density_um_inv", f"{condition}_intra_sd_density_um_inv"]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(first.size):
            row = {name: first[name][index] for name in fields[:3]}
            for condition, record in records.items():
                distribution = record["distribution"]
                row[f"{condition}_intra_mean_density_um_inv"] = distribution["intra_contribution_mean_density_um_inv"][index]
                row[f"{condition}_intra_sd_density_um_inv"] = distribution["intra_contribution_sd_density_um_inv"][index]
            writer.writerow(row)


def _write_summary_table(records: dict[str, dict[str, Any]], path: Path) -> None:
    fields = [
        "condition", "n_replicas", "peak_distance_um_mean", "peak_distance_um_sd",
        "positive_weighted_mean_distance_um_mean", "positive_weighted_mean_distance_um_sd",
        "positive_weighted_median_distance_um_mean", "positive_weighted_median_distance_um_sd",
        "replica_metric_peak_distance_um_mean", "replica_metric_peak_distance_um_sd",
        "replica_metric_positive_weighted_mean_distance_um_mean",
        "replica_metric_positive_weighted_mean_distance_um_sd",
        "replica_metric_positive_weighted_median_distance_um_mean",
        "replica_metric_positive_weighted_median_distance_um_sd",
        "group_mean_curve_peak_distance_um",
        "group_mean_curve_positive_weighted_mean_distance_um",
        "group_mean_curve_positive_weighted_median_distance_um",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for condition, record in records.items():
            summary = record["summary"]
            peak = summary["peak_distance_um"]
            mean = summary["positive_weighted_mean_distance_um"]
            median = summary["positive_weighted_median_distance_um"]
            writer.writerow({
                "condition": condition,
                "n_replicas": int(peak["n_replicas"]),
                "peak_distance_um_mean": peak["mean"],
                "peak_distance_um_sd": peak["sd"],
                "positive_weighted_mean_distance_um_mean": mean["mean"],
                "positive_weighted_mean_distance_um_sd": mean["sd"],
                "positive_weighted_median_distance_um_mean": median["mean"],
                "positive_weighted_median_distance_um_sd": median["sd"],
                "replica_metric_peak_distance_um_mean": peak["mean"],
                "replica_metric_peak_distance_um_sd": peak["sd"],
                "replica_metric_positive_weighted_mean_distance_um_mean": mean["mean"],
                "replica_metric_positive_weighted_mean_distance_um_sd": mean["sd"],
                "replica_metric_positive_weighted_median_distance_um_mean": median["mean"],
                "replica_metric_positive_weighted_median_distance_um_sd": median["sd"],
                "group_mean_curve_peak_distance_um": record["group_mean_curve_summary"]["peak_distance_um"],
                "group_mean_curve_positive_weighted_mean_distance_um": record["group_mean_curve_summary"]["positive_weighted_mean_distance_um"],
                "group_mean_curve_positive_weighted_median_distance_um": record["group_mean_curve_summary"]["positive_weighted_median_distance_um"],
            })


def _conditional_record(group_dir: Path) -> dict[str, Any]:
    """Average replica shapes already normalized per cell before aggregation."""
    edges, mids, normalized = _load_replica_density_arrays(group_dir)
    widths_um = np.diff(edges) * 1e6
    masses = np.sum(normalized * widths_um, axis=1)
    invalid = ~np.isfinite(masses) | ~np.isclose(
        masses, 1.0, rtol=1e-12, atol=1e-12
    )
    if np.any(invalid):
        indices = ", ".join(str(index) for index in np.flatnonzero(invalid))
        raise ValueError(
            f"Replica normalized Intra shapes must have signed mass 1 in "
            f"{group_dir}; invalid replica indices: {indices}"
        )
    mean = normalized.mean(axis=0)
    return {
        "x_m": mids,
        "mean": mean,
        "sd": normalized.std(axis=0, ddof=1)
        if normalized.shape[0] > 1
        else np.full(normalized.shape[1], np.nan),
        "shape_summary": _shape_summary(edges, mean),
    }


def _peak_normalized_record(group_dir: Path) -> dict[str, Any]:
    """Normalize every biological replica's positive Intra peak to one."""
    _, mids, raw = _load_replica_density_arrays(group_dir)
    peaks = np.max(raw, axis=1)
    invalid = ~np.isfinite(peaks) | (peaks <= 0)
    if np.any(invalid):
        indices = ", ".join(str(index) for index in np.flatnonzero(invalid))
        raise ValueError(
            f"Positive Intra peaks are required in {group_dir}; "
            f"invalid replica indices: {indices}"
        )
    normalized = raw / peaks[:, np.newaxis]
    return {
        "x_m": mids,
        "mean": normalized.mean(axis=0),
        "sd": normalized.std(axis=0, ddof=1)
        if normalized.shape[0] > 1
        else np.full(normalized.shape[1], np.nan),
    }


def _shape_summary(edges_m: np.ndarray, density_um_inv: np.ndarray) -> dict[str, float]:
    """Summarize one displayed mean curve with canonical bin interpolation."""
    edges = np.asarray(edges_m, dtype=float)
    density = np.asarray(density_um_inv, dtype=float)
    if edges.ndim != 1 or density.ndim != 1 or edges.size != density.size + 1:
        raise ValueError("Displayed mean curve and edges must describe the same bins")
    contribution = density * (np.diff(edges) * 1e6)
    summary = summarize_positive_intra(contribution, edges)
    return {
        "mode_nm": summary["peak_distance_um"] * 1e3,
        "mean_nm": summary["positive_weighted_mean_distance_um"] * 1e3,
        "median_nm": summary["positive_weighted_median_distance_um"] * 1e3,
    }


def _write_shape_summary_table(
    records: dict[str, dict[str, Any]], path: Path
) -> None:
    fields = ["series", "mode_nm", "mean_nm", "median_nm"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, record in records.items():
            writer.writerow({"series": name, **record["shape_summary"]})


def _write_conditional_table(records: dict[str, dict[str, Any]], path: Path) -> None:
    first = next(iter(records.values()))
    fields = ["distance_mid_m"] + [item for condition in records for item in (f"{condition}_mean_density_um_inv", f"{condition}_sd_density_um_inv")]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, value in enumerate(first["x_m"]):
            row = {"distance_mid_m": value}
            for condition, record in records.items():
                row[f"{condition}_mean_density_um_inv"] = record["mean"][index]
                row[f"{condition}_sd_density_um_inv"] = record["sd"][index]
            writer.writerow(row)


def _write_split_plots(
    target: str, records: dict[str, dict[str, Any]], output_dir: Path, stem: str,
    *, conditional: bool,
) -> None:
    """Write one full-range and one zoom comparison figure, never a dual panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"WT": "#2a78d6", "TDK": "#eb6834", "MDK": "#1baf7a"}
    first = next(iter(records.values()))
    if conditional:
        x_um = first["x_m"] * 1e6
        edges_um = np.array([0, x_um[-1] + (x_um[1] - x_um[0]) / 2])
    else:
        x_um = first["distribution"]["distance_mid_m"] * 1e6
        edges_um = np.r_[first["distribution"]["distance_left_m"][0], first["distribution"]["distance_right_m"]] * 1e6
    annotation_lines = []
    curves: list[tuple[str, np.ndarray, np.ndarray, str | None]] = []
    for condition, record in records.items():
        if conditional:
            mean, sd = record["mean"], record["sd"]
        else:
            distribution = record["distribution"]
            summary = record["summary"]
            mean = distribution["intra_contribution_mean_density_um_inv"]
            sd = distribution["intra_contribution_sd_density_um_inv"]
        curves.append((condition, mean, sd, colors.get(condition, None)))
        if not conditional:
            curve = record["group_mean_curve_summary"]
            annotation_lines.append(f"{condition} displayed mean curve: peak {curve['peak_distance_um'] * 1e3:.0f} nm; mean {curve['positive_weighted_mean_distance_um'] * 1e3:.0f} nm; median {curve['positive_weighted_median_distance_um'] * 1e3:.0f} nm")
        else:
            shape = record["shape_summary"]
            annotation_lines.append(f"{condition}: mode {shape['mode_nm']:.0f} nm; mean {shape['mean_nm']:.0f} nm; median {shape['median_nm']:.0f} nm")
    label = "Intra distribution · signed-net-mass normalized" if conditional else "Intra distribution"
    for suffix, zoom in (("full_range", False), ("zoom_0p5um", True)):
        fig, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
        for name, mean, sd, color in curves:
            axis.plot(x_um, mean, color=color, linewidth=3.2, label=name)
            axis.fill_between(x_um, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0)
        axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Distance (μm)")
        axis.set_xlim(0, min(0.5, edges_um[-1])) if zoom else axis.set_xlim(edges_um[0], edges_um[-1])
        axis.set_title("Zoom: 0–0.5 μm" if zoom else "Full range")
        axis.set_ylabel(
            "Signed-net-mass-normalized Intra density (μm⁻¹)"
            if conditional
            else "Intra contribution density (μm⁻¹)"
        )
        if zoom:
            axis.legend(title="Mean ± 1 SD", frameon=False, loc="upper right")
        fig.suptitle(f"{target} · condition comparison · {label}")
        if annotation_lines:
            fig.text(0.5, -0.04, "\n".join(annotation_lines), ha="center", va="top", fontsize=10)
        fig.savefig(output_dir / f"{stem}_{suffix}.png", dpi=180, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def _write_peak_normalized_zoom_plot(
    target: str, records: dict[str, dict[str, Any]], path: Path
) -> None:
    """Show zoomed shape curves after each biological replica's peak is set to one."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"WT": "#2a78d6", "TDK": "#eb6834", "MDK": "#1baf7a"}
    first = next(iter(records.values()))
    x_um = first["x_m"] * 1e6
    fig, axis = plt.subplots(figsize=(8.5, 5.5), constrained_layout=True)
    for name, record in records.items():
        mean, sd = record["mean"], record["sd"]
        color = colors.get(name, None)
        axis.plot(x_um, mean, color=color, linewidth=3.2, label=name)
        axis.fill_between(x_um, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0)
    axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlim(0, 0.5)
    axis.set_xlabel("Distance (μm)")
    axis.set_ylabel("Cell-balanced Intra shape / replica positive peak")
    axis.set_title("Zoom: 0–0.5 μm · peak normalized")
    axis.legend(title="Mean ± 1 SD", frameon=False, loc="upper right")
    fig.suptitle(f"{target} · condition comparison · Intra distribution · peak normalized")
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
