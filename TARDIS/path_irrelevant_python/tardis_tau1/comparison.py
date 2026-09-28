"""Cross-condition Intra comparisons for one molecular target."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


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
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_distribution_table(records, output_dir / "intra_contribution_by_distance.csv")
    _write_summary_table(records, output_dir / "intra_shape_summary.csv")
    _write_plot(target, records, output_dir / "intra_contribution.png", conditional=False)
    conditional = {condition: _conditional_record(path / "group") for condition, path in ordered.items() if condition in records}
    _write_conditional_table(conditional, output_dir / "intra_area_normalized_by_distance.csv")
    _write_plot(target, conditional, output_dir / "intra_area_normalized.png", conditional=True)


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
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_distribution_table(records, output_dir / "targets_intra_contribution_by_distance.csv")
    _write_summary_table(records, output_dir / "targets_intra_shape_summary.csv")
    _write_plot(condition, records, output_dir / "targets_intra_contribution.png", conditional=False)
    conditional_records = {target: _conditional_record(path / "group") for target, path in target_directories.items() if target in records}
    _write_conditional_table(conditional_records, output_dir / "targets_intra_area_normalized_by_distance.csv")
    _write_plot(condition, conditional_records, output_dir / "targets_intra_area_normalized.png", conditional=True)


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
    with (group_dir / "group_summary.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            summary[row["metric"]] = {key: float(value) for key, value in row.items() if key != "metric"}
    return {"distribution": distribution, "summary": summary}


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
            })


def _conditional_record(group_dir: Path) -> dict[str, Any]:
    arrays = np.load(group_dir / "group_result.npz")
    edges = arrays["edges_m"]
    widths_um = np.diff(edges) * 1e6
    raw = arrays["intra_contribution_replica_density_um_inv"]
    masses = np.sum(raw * widths_um, axis=1)
    normalized = raw / masses[:, np.newaxis]
    mean = normalized.mean(axis=0)
    return {
        "x_m": arrays["bin_mids_m"],
        "mean": mean,
        "sd": normalized.std(axis=0, ddof=1) if normalized.shape[0] > 1 else np.full(normalized.shape[1], np.nan),
        "shape_summary": _shape_summary(arrays["bin_mids_m"], mean),
    }


def _shape_summary(x_m: np.ndarray, density_um_inv: np.ndarray) -> dict[str, float]:
    """Mode, positive-density mean, and median of one displayed mean curve."""
    positive = np.maximum(np.asarray(density_um_inv, dtype=float), 0.0)
    widths_m = np.diff(np.r_[x_m[0] - (x_m[1] - x_m[0]) / 2, (x_m[:-1] + x_m[1:]) / 2, x_m[-1] + (x_m[-1] - x_m[-2]) / 2])
    masses = positive * widths_m * 1e6
    total = masses.sum()
    peak = float(x_m[np.argmax(positive)] * 1e6)
    mean = float(np.sum(masses * x_m) / total * 1e6)
    median_index = int(np.searchsorted(np.cumsum(masses), total / 2))
    return {"mode_nm": peak * 1e3, "mean_nm": mean * 1e3, "median_nm": float(x_m[median_index] * 1e9)}


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


def _write_plot(target: str, records: dict[str, dict[str, Any]], path: Path, *, conditional: bool) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.3), sharey=True, constrained_layout=True)
    annotation_lines = []
    for condition, record in records.items():
        if conditional:
            mean, sd = record["mean"], record["sd"]
        else:
            distribution = record["distribution"]
            summary = record["summary"]
            mean = distribution["intra_contribution_mean_density_um_inv"]
            sd = distribution["intra_contribution_sd_density_um_inv"]
        color = colors.get(condition, None)
        for axis in axes:
            axis.plot(x_um, mean, color=color, linewidth=3.2, label=condition)
            axis.fill_between(x_um, mean - sd, mean + sd, color=color, alpha=0.16, linewidth=0)
        if not conditional:
            annotation_lines.append(f"{condition}: peak {summary['peak_distance_um']['mean'] * 1e3:.0f} nm; mean {summary['positive_weighted_mean_distance_um']['mean'] * 1e3:.0f} nm; median {summary['positive_weighted_median_distance_um']['mean'] * 1e3:.0f} nm")
        else:
            shape = record["shape_summary"]
            annotation_lines.append(f"{condition}: mode {shape['mode_nm']:.0f} nm; mean {shape['mean_nm']:.0f} nm; median {shape['median_nm']:.0f} nm")
    for axis in axes:
        axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Distance (μm)")
    axes[0].set_xlim(edges_um[0], edges_um[-1])
    axes[0].set_title("Full range")
    axes[0].set_ylabel("Intra contribution density (μm⁻¹)")
    axes[1].set_xlim(0, min(0.5, edges_um[-1]))
    axes[1].set_title("Zoom: 0–0.5 μm")
    axes[1].legend(title="Mean ± 1 SD", frameon=False, loc="upper right")
    label = "Intra distribution · area normalized" if conditional else "Intra distribution"
    fig.suptitle(f"{target} · condition comparison · {label}", y=1.04)
    if annotation_lines:
        fig.text(0.5, -0.04, "\n".join(annotation_lines), ha="center", va="top", fontsize=10)
    fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)
