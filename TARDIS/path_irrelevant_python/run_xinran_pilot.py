from __future__ import annotations

import argparse
from pathlib import Path

from tardis_tau1 import Tau1Config, analyze_cell
from tardis_tau1.outputs import save_result

DEFAULT_INPUT = Path(
    "Xinran_ComparisonData/Xinran_ComparisonData/"
    "Miro1/Miro1_WT/20251016_WT_Miro1_V7.mat"
)
DEFAULT_RECORD = "C2-101625_WT_Miro1_T1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SI-calibrated tau=1 analysis for one Xinran Base cell."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--record", default=DEFAULT_RECORD)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("path_irrelevant_python/pilot_output"),
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Tau1Config(save_plots=not args.no_plot, bootstrap_count=0)
    result = analyze_cell(
        args.input,
        ids={
            "condition": "WT",
            "target": "Miro1",
            "replicate": "20251016_WT_Miro1_V7",
            "cell": args.record,
            "position_variable": "data",
            "record_name": args.record,
            "coordinate_unit": "pixel",
            "pixel_size_m": 117e-9,
            "frame_time_s": 0.02,
        },
        config=config,
    )
    destination = (
        args.output
        / "WT"
        / "Miro1"
        / "20251016_WT_Miro1_V7"
        / args.record
    )
    save_result(result, destination, config, "cell")
    print(f"Saved pilot result to {destination.resolve()}")
    print(
        f"localizations={result.counts['localizations']}; "
        f"frames={result.counts['frames']}; "
        f"beta={result.distributions['beta']:.12g}; "
        f"analysis_s={result.provenance['execution']['analysis_runtime_seconds']:.3f}; "
        f"sha256={result.source['file_sha256'][:12]}; "
        f"qc={result.qc['status']}"
    )


if __name__ == "__main__":
    main()
