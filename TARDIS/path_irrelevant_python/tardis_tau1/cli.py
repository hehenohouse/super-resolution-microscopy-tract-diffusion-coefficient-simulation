from __future__ import annotations

import argparse

from .batch import run_batch
from .config import Tau1Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute tau=1 Total/Inter/Intra distributions by cell and genotype."
    )
    parser.add_argument("manifest", help="CSV manifest with one row per cell")
    parser.add_argument("output", help="Output directory")
    parser.add_argument("--max-distance-um", type=float, default=3.0)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--background-tail-start", type=float, default=0.7)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--bootstrap-count", type=int, default=2_000)
    parser.add_argument("--bootstrap-seed", type=int, default=1_729)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Tau1Config(
        max_distance_m=args.max_distance_um * 1e-6,
        n_bins=args.bins,
        min_distance_background_fit=args.background_tail_start,
        chunk_size=args.chunk_size,
        bootstrap_count=args.bootstrap_count,
        bootstrap_seed=args.bootstrap_seed,
        save_plots=not args.no_plots,
    )
    result = run_batch(args.manifest, args.output, config)
    print(
        f"Processed {result.successful_cells}/{result.enabled_cells} cells; "
        f"generated {len(result.genotype_results)} genotype result(s); "
        f"failures={result.failed_cells}."
    )
    if result.failures:
        print(f"Failure details: {result.output_root}/failures.csv")


if __name__ == "__main__":
    main()
