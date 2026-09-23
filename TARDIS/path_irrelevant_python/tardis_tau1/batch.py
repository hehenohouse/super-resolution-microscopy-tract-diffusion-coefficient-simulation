from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Tau1Config
from .core import CellResult, GenotypeResult, aggregate_genotype, analyze_cell
from .outputs import safe_component, save_result

_REQUIRED_COLUMNS = {
    "condition_id",
    "target_id",
    "genotype_id",
    "replicate_id",
    "cell_id",
    "input_file",
}


@dataclass
class BatchResult:
    manifest_path: str
    output_root: str
    enabled_cells: int
    successful_cells: int
    failed_cells: int
    cell_results: list[CellResult]
    genotype_results: list[GenotypeResult]
    failures: list[dict[str, Any]]


def run_batch(
    manifest_path: str | Path,
    output_root: str | Path,
    config: Tau1Config | None = None,
) -> BatchResult:
    """Run cell analyses and equal-cell genotype aggregation from a CSV manifest."""
    config = config or Tau1Config()
    manifest_path = Path(manifest_path).resolve()
    output_root = Path(output_root).resolve()
    rows = read_manifest(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)

    cell_results: list[CellResult] = []
    failures: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[CellResult]] = defaultdict(list)
    enabled_count = 0
    for row_number, row in enumerate(rows, start=2):
        if not _parse_bool(row.get("enabled", "true")):
            continue
        enabled_count += 1
        input_path = Path(row["input_file"])
        if not input_path.is_absolute():
            input_path = manifest_path.parent / input_path
        ids = {
            "condition": row["condition_id"],
            "target": row["target_id"],
            "genotype": row["genotype_id"],
            "replicate": row["replicate_id"],
            "cell": row["cell_id"],
            "position_variable": row.get("position_variable") or "pos",
            "coordinate_unit": row.get("coordinate_unit") or "m",
            "frame_time_s": row.get("frame_time_s") or None,
        }
        try:
            result = analyze_cell(input_path, ids, config)
            cell_results.append(result)
            key = (ids["condition"], ids["target"], ids["genotype"])
            grouped[key].append(result)
            cell_dir = (
                output_root
                / "cells"
                / safe_component(ids["condition"])
                / safe_component(ids["target"])
                / safe_component(ids["genotype"])
                / safe_component(ids["replicate"])
                / safe_component(ids["cell"])
            )
            save_result(result, cell_dir, config, "cell")
        except Exception as exc:  # batch must continue and report every failed cell
            failures.append(
                {
                    "row": row_number,
                    "condition": ids["condition"],
                    "target": ids["target"],
                    "genotype": ids["genotype"],
                    "replicate": ids["replicate"],
                    "cell": ids["cell"],
                    "message": str(exc),
                }
            )

    genotype_results: list[GenotypeResult] = []
    for (condition, target, genotype), cells in grouped.items():
        result = aggregate_genotype(cells, config)
        matching_failures = [
            failure
            for failure in failures
            if (
                failure["condition"],
                failure["target"],
                failure["genotype"],
            )
            == (condition, target, genotype)
        ]
        result.cells["excluded_count"] = len(matching_failures)
        result.cells["excluded_cells"] = matching_failures
        genotype_results.append(result)
        genotype_dir = (
            output_root
            / "genotypes"
            / safe_component(condition)
            / safe_component(target)
            / safe_component(genotype)
        )
        save_result(result, genotype_dir, config, "genotype")
        _write_cell_summary(cells, genotype_dir / "cell_summary.csv")

    _write_failures(failures, output_root / "failures.csv")
    return BatchResult(
        manifest_path=str(manifest_path),
        output_root=str(output_root),
        enabled_cells=enabled_count,
        successful_cells=len(cell_results),
        failed_cells=len(failures),
        cell_results=cell_results,
        genotype_results=genotype_results,
        failures=failures,
    )


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read manifest rows and reject missing columns or duplicate cell keys."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(_REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"Missing manifest columns: {', '.join(missing)}")
        rows = list(reader)
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(
            row[name]
            for name in (
                "condition_id",
                "target_id",
                "genotype_id",
                "replicate_id",
                "cell_id",
            )
        )
        if key in seen:
            raise ValueError(f"Duplicate composite cell identifier: {key}")
        seen.add(key)
    return rows


def _write_cell_summary(cells: list[CellResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cell_id",
        "replicate_id",
        "beta",
        "localizations",
        "total_pairs",
        "inter_pairs",
        "negative_intra_bins",
        "qc_status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "cell_id": cell.ids["cell"],
                    "replicate_id": cell.ids["replicate"],
                    "beta": cell.distributions["beta"],
                    "localizations": cell.counts["localizations"],
                    "total_pairs": cell.counts["total_retained_pairs"],
                    "inter_pairs": cell.counts["inter_retained_pairs"],
                    "negative_intra_bins": cell.qc["metrics"]["negative_intra_bins"],
                    "qc_status": cell.qc["status"],
                }
            )


def _write_failures(failures: list[dict[str, Any]], path: Path) -> None:
    fields = ["row", "condition", "target", "genotype", "replicate", "cell", "message"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(failures)


def _parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "true").strip().lower() in {"true", "1", "yes", "y"}
