"""Run cell-balanced TARDIS replicas and a replica-balanced group analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from tardis_tau1 import Tau1Config, aggregate_group, analyze_replica, save_group_result
from tardis_tau1.outputs import save_replica_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze matched Xinran replicas and calculate a group mean with t-based uncertainty."
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Tau1Config(n_bins=args.bins, save_plots=not args.no_plot, bootstrap_count=0)
    replicas = []
    for path in args.input:
        result, cells = analyze_replica(
            path,
            condition=args.condition,
            target=args.target,
            replica=path.stem,
            frame_interval_s=0.02,
            pixel_size_m=117e-9,
            config=config,
        )
        save_replica_result(result, cells, args.output / "replicas" / path.stem, config)
        replicas.append(result)
        print(f"Replica {path.stem}: {result.counts['n_cells']} cells")

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
