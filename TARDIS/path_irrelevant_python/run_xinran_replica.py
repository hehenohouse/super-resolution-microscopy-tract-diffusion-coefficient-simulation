from __future__ import annotations

import argparse
from pathlib import Path

from tardis_tau1 import Tau1Config, analyze_replica
from tardis_tau1.outputs import save_replica_result

DEFAULT_INPUT = Path(
    "Xinran_ComparisonData/Xinran_ComparisonData/"
    "Miro1/Miro1_WT/20251016_WT_Miro1_V7.mat"
)
DEFAULT_REPLICA = "20251016_WT_Miro1_V7"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze all Base cells in one SI-calibrated Xinran tau=1 replica."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output", type=Path, default=Path("path_irrelevant_python/replica_output")
    )
    parser.add_argument("--condition", default="WT")
    parser.add_argument("--target", default="Miro1")
    parser.add_argument("--replica", default=DEFAULT_REPLICA)
    parser.add_argument("--expected-cells", type=int, default=10)
    parser.add_argument("--bins", type=int, default=300)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Tau1Config(
        n_bins=args.bins, save_plots=not args.no_plot, bootstrap_count=0
    )
    result, cells = analyze_replica(
        args.input,
        condition=args.condition,
        target=args.target,
        replica=args.replica,
        frame_interval_s=0.02,
        pixel_size_m=117e-9,
        config=config,
    )
    if result.cells["discovered_count"] != args.expected_cells:
        raise ValueError(
            f"Expected {args.expected_cells} Base cells, discovered "
            f"{result.cells['discovered_count']}"
        )
    destination = args.output / args.condition / args.target / args.replica
    save_replica_result(result, cells, destination, config)
    print(f"Saved replica result to {destination.resolve()}")
    print(
        f"cells={result.counts['n_cells']}; "
        f"total_pairs={result.counts['total_retained_pairs']}; "
        f"background_pairs={result.counts['inter_retained_pairs']}; "
        f"beta={result.distributions['beta']:.12g}; "
        f"analysis_s={result.provenance['execution']['analysis_runtime_seconds']:.3f}; "
        f"sha256={result.source['file_sha256'][:12]}; "
        f"qc={result.qc['status']}"
    )


if __name__ == "__main__":
    main()
