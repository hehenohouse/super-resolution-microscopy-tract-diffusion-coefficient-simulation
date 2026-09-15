from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .simulator import SimulationResult


def save_animation(
    result: SimulationResult,
    output_path: str | Path,
    *,
    dpi: int = 110,
) -> Path:
    """Save a side-by-side ground-truth and observation animation."""
    if result.bounds_um.shape != (2, 2):
        raise ValueError("The current animation renderer requires two dimensions")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, (truth_axis, observed_axis) = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        facecolor="#0b1020",
        constrained_layout=True,
    )

    x_bounds, y_bounds = result.bounds_um
    for axis in (truth_axis, observed_axis):
        axis.set_facecolor("#050812")
        axis.set_xlim(*x_bounds)
        axis.set_ylim(*y_bounds)
        axis.set_aspect("equal")
        axis.set_xlabel("x (µm)", color="#d9e2f2")
        axis.set_ylabel("y (µm)", color="#d9e2f2")
        axis.tick_params(colors="#a8b3c7")
        for spine in axis.spines.values():
            spine.set_color("#38445a")

    truth_axis.set_title("Ground truth", color="white")
    observed_axis.set_title("Observation available to tracking", color="white")

    empty = np.empty((0, 2), dtype=np.float64)
    truth_non_emitting = truth_axis.scatter(
        [],
        [],
        s=16,
        c="#64748b",
        alpha=0.42,
        linewidths=0,
        label="active, non-emitting",
    )
    truth_emitting = truth_axis.scatter(
        [],
        [],
        s=24,
        c="#4de3ff",
        alpha=0.95,
        linewidths=0,
        label="emitting",
    )
    observed_glow = observed_axis.scatter(
        [], [], s=105, c="#4de3ff", alpha=0.14, linewidths=0
    )
    observed_points = observed_axis.scatter(
        [], [], s=24, c="#d9fbff", alpha=1.0, linewidths=0
    )
    truth_axis.legend(
        loc="upper right",
        facecolor="#111827",
        edgecolor="#38445a",
        labelcolor="white",
    )

    status_text = figure.suptitle("", color="white", fontsize=11)

    def update(frame: int):
        frame_positions = result.positions_um[frame]
        frame_emitting = result.emitting[frame]
        frame_active = result.active[frame]
        frame_visible = result.visible[frame]
        non_emitting = frame_active & ~frame_emitting
        inactive = ~frame_active

        truth_non_emitting.set_offsets(
            frame_positions[non_emitting] if np.any(non_emitting) else empty
        )
        truth_emitting.set_offsets(
            frame_positions[frame_emitting] if np.any(frame_emitting) else empty
        )
        observed_frame = result.observed_positions_um[frame]
        visible_positions = (
            observed_frame[frame_visible] if np.any(frame_visible) else empty
        )
        observed_glow.set_offsets(visible_positions)
        observed_points.set_offsets(visible_positions)
        status_text.set_text(
            f"{result.scenario.scenario_id}   |   "
            f"t = {result.time_s[frame]:.2f} s   |   "
            f"emitting {frame_emitting.sum()}   "
            f"non-emitting {non_emitting.sum()}   "
            f"inactive {inactive.sum()}   visible {frame_visible.sum()}"
        )
        return (
            truth_non_emitting,
            truth_emitting,
            observed_glow,
            observed_points,
            status_text,
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=result.config.n_frames,
        interval=result.config.frame_interval_s * 1000.0,
        blit=False,
    )
    frames_per_second = max(1, round(1.0 / result.config.frame_interval_s))
    animation.save(
        output_path,
        writer=PillowWriter(fps=frames_per_second),
        dpi=dpi,
    )
    plt.close(figure)
    return output_path
