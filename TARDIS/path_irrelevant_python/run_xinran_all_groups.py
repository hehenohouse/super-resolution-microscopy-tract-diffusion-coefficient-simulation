"""Batch-run replica-balanced TARDIS analyses for every target and condition."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tardis_tau1 import Tau1Config, aggregate_group, analyze_replica, save_group_result
from tardis_tau1.comparison import save_condition_target_comparison, save_target_condition_comparison
from tardis_tau1.outputs import save_replica_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-run all Xinran condition × target TARDIS group analyses."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=300)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Tau1Config(n_bins=args.bins, save_plots=not args.no_plot, bootstrap_count=0)
    root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    target_outputs: dict[str, dict[str, Path]] = {}
    for target_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        target = target_dir.name
        completed_conditions: dict[str, Path] = {}
        for input_dir in sorted(path for path in target_dir.iterdir() if path.is_dir()):
            prefix = f"{target}_"
            if not input_dir.name.startswith(prefix):
                continue
            condition = input_dir.name.removeprefix(prefix)
            inputs = sorted(input_dir.glob("*.mat"))
            destination = output_root / target / condition
            if not inputs:
                rows.append(_row(target, condition, "skipped_no_input", 0, 0, destination, ""))
                continue
            try:
                replicas = []
                for path in inputs:
                    result, cells = analyze_replica(
                        path,
                        condition=condition,
                        target=target,
                        replica=path.stem,
                        frame_interval_s=0.02,
                        pixel_size_m=117e-9,
                        config=config,
                    )
                    save_replica_result(
                        result, cells, destination / "replicas" / path.stem, config
                    )
                    replicas.append(result)
                group = aggregate_group(replicas)
                save_group_result(group, destination / "group", save_plots=not args.no_plot)
                cells = sum(result.counts["n_cells"] for result in replicas)
                rows.append(_row(target, condition, "complete", len(replicas), cells, destination, ""))
                completed_conditions[condition] = destination
                print(f"Complete {target}/{condition}: {len(replicas)} replicas, {cells} cells")
            except Exception as exc:
                rows.append(_row(target, condition, "failed", 0, 0, destination, str(exc)))
                print(f"Failed {target}/{condition}: {exc}")
        save_target_condition_comparison(target, completed_conditions, output_root / target / "comparison")
        target_outputs[target] = completed_conditions
    conditions = sorted({condition for groups in target_outputs.values() for condition in groups})
    for condition in conditions:
        target_groups = {
            target: groups[condition]
            for target, groups in target_outputs.items()
            if condition in groups
        }
        save_condition_target_comparison(
            condition, target_groups, output_root / "condition_comparison" / condition
        )
    summary_path = output_root / "batch_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    complete = sum(row["status"] == "complete" for row in rows)
    print(f"Batch complete: {complete}/{len(rows)} groups; summary={summary_path}")


def _row(
    target: str,
    condition: str,
    status: str,
    replicas: int,
    cells: int,
    output: Path,
    message: str,
) -> dict[str, object]:
    return {
        "target": target,
        "condition": condition,
        "status": status,
        "replicas": replicas,
        "cells": cells,
        "output_directory": str(output),
        "message": message,
    }


if __name__ == "__main__":
    main()
