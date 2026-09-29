"""Run cell-balanced TARDIS replicas and a replica-balanced group analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from tardis_tau1 import Tau1Config, aggregate_group, analyze_replica, save_group_result
from tardis_tau1.cache import (
    build_replica_identity,
    load_cached_replica,
    write_cache_manifest,
)
from tardis_tau1.outputs import save_replica_result
from tardis_tau1.provenance import fingerprint_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze matched Xinran replicas with equal replica weights, sample "
            "SD/SEM, and mean ± 1 SD plot bands."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="MATLAB v7.3 replica file. Supply once per biological replicate.",
    )
    parser.add_argument("--output", type=Path, default=Path("path_irrelevant_python/group_output"))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--bins", type=int, default=300)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Tau1Config(n_bins=args.bins, save_plots=not args.no_plot, bootstrap_count=0)
    replicas = []
    for path in args.input:
        path = path.resolve()
        destination = args.output / "replicas" / path.stem
        fingerprint = fingerprint_file(path)
        identity = build_replica_identity(
            fingerprint=fingerprint,
            condition=args.condition,
            target=args.target,
            replica=path.stem,
            variable_name="data",
            frame_interval_s=0.02,
            pixel_size_m=117e-9,
            config=config,
        )
        result = None if args.force_recompute else load_cached_replica(
            destination, identity, save_plots=config.save_plots
        )
        if result is None:
            result, cells = analyze_replica(
                path,
                condition=args.condition,
                target=args.target,
                replica=path.stem,
                frame_interval_s=0.02,
                pixel_size_m=117e-9,
                config=config,
                fingerprint=fingerprint,
            )
            save_replica_result(result, cells, destination, config)
            write_cache_manifest(
                destination, identity, result, save_plots=config.save_plots
            )
            cache_status = "computed"
        else:
            cache_status = "cache hit"
        replicas.append(result)
        print(
            f"Replica {path.stem}: {result.counts['n_cells']} cells; "
            f"{cache_status}"
        )

    group = aggregate_group(replicas)
    destination = args.output / "group"
    save_group_result(group, destination, save_plots=not args.no_plot)
    print(f"Saved group result to {destination.resolve()}")
    print(
        f"replicas={len(group.replica_ids)}; "
        f"error_model={group.parameters['error_model']}"
    )


if __name__ == "__main__":
    main()
