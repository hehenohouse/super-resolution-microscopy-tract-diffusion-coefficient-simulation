"""Biological-replicate aggregation for τ=1 TARDIS distributions."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .core import ReplicaResult


_COMPONENTS = ("total", "inter_contribution", "intra_contribution")
_SUMMARY_METRICS = (
    "beta",
    "total_positive_mass",
    "positive_mass_0_300_nm",
    "peak_distance_um",
    "peak_density_um_inv",
    "positive_weighted_mean_distance_um",
    "positive_weighted_median_distance_um",
)


@dataclass
class GroupResult:
    """Equal-biological-replicate group result with bin-wise uncertainty."""

    ids: dict[str, str]
    replica_ids: list[str]
    parameters: dict[str, Any]
    distributions: dict[str, dict[str, np.ndarray]]
    replica_density_distributions: dict[str, np.ndarray]
    summary_statistics: dict[str, dict[str, float]]
    replica_summaries: list[dict[str, Any]]
    schema_version: str = "tau1-group-replica-balanced-v1"


def aggregate_group(replicas: Iterable[ReplicaResult]) -> GroupResult:
    """Average matched replicas equally and calculate sample SD and SEM."""
    items = list(replicas)
    if not items:
        raise ValueError("At least one replica result is required")
    first = items[0]
    reference_ids = {key: first.ids[key] for key in ("condition", "target")}
    edges = np.asarray(first.parameters["edges_m"], dtype=float)
    for replica in items:
        if {key: replica.ids[key] for key in reference_ids} != reference_ids:
            raise ValueError("All replicas must have the same condition and target")
        if not np.array_equal(np.asarray(replica.parameters["edges_m"]), edges):
            raise ValueError("All replicas must use identical distance bins")

    widths_um = np.diff(edges) * 1e6
    distributions: dict[str, dict[str, np.ndarray]] = {}
    replica_density_distributions: dict[str, np.ndarray] = {}
    for component in _COMPONENTS:
        stack = np.stack(
            [replica.distributions[component] / widths_um for replica in items]
        )
        mean, sd, sem, lower, upper = _mean_and_error_band(stack)
        distributions[component] = {
            "mean_density_um_inv": mean,
            "sd_density_um_inv": sd,
            "sem_density_um_inv": sem,
            "mean_minus_sd_density_um_inv": lower,
            "mean_plus_sd_density_um_inv": upper,
        }
        replica_density_distributions[component] = stack

    replica_summaries = [_replica_summary(replica) for replica in items]
    summary_statistics = {
        metric: _scalar_statistics(
            np.asarray([row[metric] for row in replica_summaries], dtype=float),
        )
        for metric in _SUMMARY_METRICS
    }
    return GroupResult(
        ids=reference_ids,
        replica_ids=[replica.ids["replicate"] for replica in items],
        parameters={
            "edges_m": edges,
            "bin_mids_m": (edges[:-1] + edges[1:]) / 2,
            "n_replicas": len(items),
            "replica_aggregation": "equal_weight",
            "error_model": "sample_sd_sem_and_mean_plus_minus_sd_across_biological_replicates",
        },
        distributions=distributions,
        replica_density_distributions=replica_density_distributions,
        summary_statistics=summary_statistics,
        replica_summaries=replica_summaries,
    )


def save_group_result(result: GroupResult, output_dir: str | Path, *, save_plots: bool) -> None:
    """Write group arrays, tables, provenance metadata, and optional figures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_group_distributions(result, output / "group_mean_distributions.csv")
    _write_group_summary(result, output / "group_summary.csv")
    _write_replica_summaries(result, output / "replica_summary.csv")
    arrays: dict[str, np.ndarray] = {
        "edges_m": np.asarray(result.parameters["edges_m"]),
        "bin_mids_m": np.asarray(result.parameters["bin_mids_m"]),
        "replica_ids": np.asarray(result.replica_ids),
    }
    for component, statistics in result.distributions.items():
        for name, values in statistics.items():
            arrays[f"{component}_{name}"] = np.asarray(values)
        arrays[f"{component}_replica_density_um_inv"] = np.asarray(
            result.replica_density_distributions[component]
        )
    np.savez_compressed(output / "group_result.npz", **arrays)
    (output / "group_result.json").write_text(
        json.dumps(_json_safe(asdict(result)), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if save_plots:
        _write_group_distribution_plots(result, output)
        _write_replica_intra_overlays(result, output)


def _mean_and_error_band(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    if values.shape[0] < 2:
        nan = np.full_like(mean, np.nan, dtype=float)
        return mean, nan, nan, nan, nan
    sd = values.std(axis=0, ddof=1)
    sem = sd / np.sqrt(values.shape[0])
    return mean, sd, sem, mean - sd, mean + sd


def _scalar_statistics(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    mean = float(np.mean(finite)) if finite.size else float("nan")
    if finite.size < 2:
        return {
            "n_replicas": int(finite.size),
            "mean": mean,
            "sd": float("nan"),
            "sem": float("nan"),
            "mean_minus_sd": float("nan"),
            "mean_plus_sd": float("nan"),
        }
    sd = float(np.std(finite, ddof=1))
    sem = sd / np.sqrt(finite.size)
    return {
        "n_replicas": int(finite.size),
        "mean": mean,
        "sd": sd,
        "sem": float(sem),
        "mean_minus_sd": float(mean - sd),
        "mean_plus_sd": float(mean + sd),
    }


def _replica_summary(replica: ReplicaResult) -> dict[str, Any]:
    positive = replica.summaries["positive_intra"]
    return {
        "replica_id": replica.ids["replicate"],
        "n_cells": replica.counts["n_cells"],
        "beta": replica.distributions["beta"],
        **{name: positive[name] for name in _SUMMARY_METRICS if name != "beta"},
    }


def _write_group_distributions(result: GroupResult, path: Path) -> None:
    edges = np.asarray(result.parameters["edges_m"])
    mids = np.asarray(result.parameters["bin_mids_m"])
    fields = ["distance_left_m", "distance_right_m", "distance_mid_m"]
    for component in _COMPONENTS:
        fields.extend(
            f"{component}_{name}"
            for name in (
                "mean_density_um_inv",
                "sd_density_um_inv",
                "sem_density_um_inv",
                "mean_minus_sd_density_um_inv",
                "mean_plus_sd_density_um_inv",
            )
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(mids.size):
            row: dict[str, Any] = {
                "distance_left_m": edges[index],
                "distance_right_m": edges[index + 1],
                "distance_mid_m": mids[index],
            }
            for component in _COMPONENTS:
                for name, values in result.distributions[component].items():
                    row[f"{component}_{name}"] = values[index]
            writer.writerow(row)


def _write_group_summary(result: GroupResult, path: Path) -> None:
    fields = ["metric", "n_replicas", "mean", "sd", "sem", "mean_minus_sd", "mean_plus_sd"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metric, statistics in result.summary_statistics.items():
            writer.writerow({"metric": metric, **statistics})


def _write_replica_summaries(result: GroupResult, path: Path) -> None:
    fields = list(result.replica_summaries[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result.replica_summaries)


def _write_group_distribution_plots(result: GroupResult, output_dir: Path) -> None:
    plt = _pyplot()
    x_um = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    edges_um = np.asarray(result.parameters["edges_m"]) * 1e6
    styles = {
        "total": ("#2a78d6", "Total"),
        "inter_contribution": ("#eb6834", "Inter"),
        "intra_contribution": ("#1baf7a", "Intra"),
    }
    title = f"{result.ids['condition']} · {result.ids['target']} · n={len(result.replica_ids)} biological replicas · mean ± 1 SD"
    for suffix, zoom in (("full_range", False), ("zoom_0p5um", True)):
        fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        if zoom:
            intra = result.distributions["intra_contribution"]
            axis.plot(x_um, intra["mean_density_um_inv"], color="#1baf7a", linewidth=3.2, label="Intra")
            axis.fill_between(x_um, intra["mean_minus_sd_density_um_inv"], intra["mean_plus_sd_density_um_inv"], color="#1baf7a", alpha=0.20, linewidth=0, label="± 1 SD")
        else:
            for component in _COMPONENTS:
                color, label = styles[component]
                stats = result.distributions[component]
                axis.plot(x_um, stats["mean_density_um_inv"], color=color, linewidth=3, label=label)
                axis.fill_between(x_um, stats["mean_minus_sd_density_um_inv"], stats["mean_plus_sd_density_um_inv"], color=color, alpha=0.16, linewidth=0)
        axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Distance (μm)")
        axis.set_xlim(0, min(0.5, edges_um[-1])) if zoom else axis.set_xlim(edges_um[0], edges_um[-1])
        axis.set_ylabel("Mean probability density (μm⁻¹)")
        axis.set_title("Zoom: 0–0.5 μm · Intra only" if zoom else "Full range")
        if zoom:
            axis.legend(frameon=False, loc="upper right")
        fig.suptitle(title)
        fig.savefig(output_dir / f"group_mean_distributions_{suffix}.png", dpi=180, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def _write_replica_intra_overlays(result: GroupResult, output_dir: Path) -> None:
    plt = _pyplot()
    x_um = np.asarray(result.parameters["bin_mids_m"]) * 1e6
    edges_um = np.asarray(result.parameters["edges_m"]) * 1e6
    colors = plt.get_cmap("tab10").colors
    for suffix, zoom in (("full_range", False), ("zoom_0p5um", True)):
        fig, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        stats = result.distributions["intra_contribution"]
        for index, (replica_id, values) in enumerate(zip(
            result.replica_ids,
            result.replica_density_distributions["intra_contribution"],
        )):
            axis.plot(
                x_um,
                values,
                color=colors[index % len(colors)],
                linewidth=2.0,
                alpha=0.80,
                label=replica_id,
            )
        axis.plot(x_um, stats["mean_density_um_inv"], color="#1baf7a", linewidth=3.5, label="Group mean Intra")
        axis.fill_between(
            x_um,
            stats["mean_minus_sd_density_um_inv"],
            stats["mean_plus_sd_density_um_inv"],
            color="#1baf7a",
            alpha=0.20,
            linewidth=0,
            label="± 1 SD",
        )
        axis.axhline(0, color="#777777", linewidth=0.8, linestyle=":")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("Distance (μm)")
        axis.set_xlim(0, min(0.5, edges_um[-1])) if zoom else axis.set_xlim(edges_um[0], edges_um[-1])
        axis.set_ylabel("Intra contribution density (μm⁻¹)")
        axis.set_title("Zoom: 0–0.5 μm" if zoom else "Full range")
        if zoom:
            axis.legend(frameon=False, loc="upper right")
        fig.savefig(output_dir / f"replica_intra_overlay_{suffix}.png", dpi=180, facecolor="white", bbox_inches="tight")
        plt.close(fig)


def _pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
