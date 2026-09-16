from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_GROUND_TRUTH = Path(
    "outputs/multi_chromosome_bound_dynamic_2d/ground_truth.npz"
)
DEFAULT_LAG_TIMES_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare mean absolute protein displacement along chromosome-local "
            "major and minor axes."
        )
    )
    parser.add_argument(
        "ground_truth",
        nargs="?",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
        help="Path to a multi-chromosome ground_truth.npz file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.ground_truth, allow_pickle=False) as ground_truth:
        if "chromosome_local_positions_um" not in ground_truth.files:
            raise ValueError(
                "ground truth does not contain chromosome-local positions"
            )
        local_positions = ground_truth["chromosome_local_positions_um"]
        frame_times_s = ground_truth["frame_times_s"]

    if local_positions.ndim != 3 or local_positions.shape[2] != 2:
        raise ValueError(
            "chromosome_local_positions_um must have shape (frames, particles, 2)"
        )
    if frame_times_s.size < 2:
        raise ValueError("at least two frames are required")

    frame_interval_s = float(frame_times_s[1] - frame_times_s[0])
    print(f"Ground truth: {args.ground_truth}")
    print(f"Frames: {local_positions.shape[0]}")
    print(f"Particles: {local_positions.shape[1]}")
    print(f"Frame interval: {frame_interval_s:.3f} s")
    print()
    print("Mean absolute displacement in chromosome-local coordinates")
    print("lag_s   major_um   minor_um   major/minor")

    for requested_lag_s in DEFAULT_LAG_TIMES_S:
        lag_frames = max(1, round(requested_lag_s / frame_interval_s))
        if lag_frames >= local_positions.shape[0]:
            continue
        displacement = (
            local_positions[lag_frames:] - local_positions[:-lag_frames]
        )
        mean_absolute = np.mean(np.abs(displacement), axis=(0, 1))
        ratio = float(mean_absolute[0] / mean_absolute[1])
        print(
            f"{lag_frames * frame_interval_s:5.2f}   "
            f"{mean_absolute[0]:8.4f}   "
            f"{mean_absolute[1]:8.4f}   "
            f"{ratio:11.3f}"
        )

    per_particle_range = np.ptp(local_positions, axis=0)
    mean_range = per_particle_range.mean(axis=0)
    print()
    print("Mean per-particle coordinate range over the complete recording")
    print(f"major axis: {mean_range[0]:.4f} um")
    print(f"minor axis: {mean_range[1]:.4f} um")
    print(f"major/minor: {mean_range[0] / mean_range[1]:.3f}")


if __name__ == "__main__":
    main()
