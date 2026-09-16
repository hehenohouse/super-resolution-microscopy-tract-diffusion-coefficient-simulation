from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse

from .simulator import SimulationResult


def _chromosome_preview_data(
    result: SimulationResult,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[float, float],
] | None:
    parameters = result.scenario.boundary.parameters()
    if not {
        "semi_major_axis_um",
        "semi_minor_axis_um",
    }.issubset(parameters):
        return None
    if (
        result.scene_instance_centers_um is not None
        and result.scene_instance_orientations_rad is not None
        and result.particle_scene_ids is not None
    ):
        centers = result.scene_instance_centers_um
        orientations = result.scene_instance_orientations_rad
        assignments = result.particle_scene_ids
    elif (
        result.scene_centers_um is not None
        and result.scene_orientations_rad is not None
    ):
        centers = result.scene_centers_um[:, None, :]
        orientations = result.scene_orientations_rad[:, None]
        assignments = np.zeros(result.config.n_particles, dtype=np.int64)
    else:
        return None
    axes = (
        float(parameters["semi_major_axis_um"]),
        float(parameters["semi_minor_axis_um"]),
    )
    return centers, orientations, assignments, axes


def _protein_trail_data(
    result: SimulationResult,
    *,
    max_particles_per_chromosome: int = 2,
    duration_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Select deterministic private-truth protein trails for chromosome scenes."""
    chromosome_data = _chromosome_preview_data(result)
    if chromosome_data is None:
        return None
    if max_particles_per_chromosome <= 0:
        raise ValueError("max_particles_per_chromosome must be positive")
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")

    _, _, assignments, _ = chromosome_data
    selected_particle_ids: list[int] = []
    selected_chromosome_ids: list[int] = []
    chromosome_count = int(assignments.max()) + 1 if assignments.size else 0
    for chromosome_id in range(chromosome_count):
        particle_ids = np.flatnonzero(assignments == chromosome_id)
        selected = particle_ids[:max_particles_per_chromosome]
        selected_particle_ids.extend(int(particle_id) for particle_id in selected)
        selected_chromosome_ids.extend([chromosome_id] * selected.size)

    trail_frame_count = max(1, round(duration_s / result.config.frame_interval_s))
    return (
        np.asarray(selected_particle_ids, dtype=np.int64),
        np.asarray(selected_chromosome_ids, dtype=np.int64),
        trail_frame_count,
    )


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

    truth_title = "Ground truth inside observation window"
    if result.scenario.observation.input_dimension != (
        result.scenario.observation.output_dimension
    ):
        truth_title = "Ground truth in observation volume (XY projection)"
    truth_axis.set_title(truth_title, color="white")
    observed_axis.set_title("Observation available to tracking", color="white")

    empty = np.empty((0, 2), dtype=np.float64)
    truth_non_emitting = truth_axis.scatter(
        [],
        [],
        s=16,
        c="#64748b",
        alpha=0.42,
        linewidths=0,
        label="in FOV, non-emitting",
        zorder=4,
    )
    truth_emitting = truth_axis.scatter(
        [],
        [],
        s=24,
        c="#4de3ff",
        alpha=0.95,
        linewidths=0,
        label="in FOV, emitting",
        zorder=5,
    )
    observed_glow = observed_axis.scatter(
        [], [], s=105, c="#4de3ff", alpha=0.14, linewidths=0
    )
    observed_points = observed_axis.scatter(
        [], [], s=24, c="#d9fbff", alpha=1.0, linewidths=0
    )

    chromosome_data = _chromosome_preview_data(result)
    chromosome_ellipses: list[Ellipse] = []
    chromosome_centers = []
    chromosome_directions = []
    chromosome_trails = []
    protein_trails = []
    protein_trail_particle_ids = np.empty(0, dtype=np.int64)
    protein_trail_frame_count = 1
    trail_frame_count = max(
        1, round(2.0 / result.config.frame_interval_s)
    )
    protein_trail_data = _protein_trail_data(result)
    if chromosome_data is not None:
        scene_centers, _, _, ellipse_axes = chromosome_data
        color_map = plt.get_cmap("tab10")
        for chromosome_id in range(scene_centers.shape[1]):
            color = color_map(chromosome_id % 10)
            ellipse = Ellipse(
                (0.0, 0.0),
                width=2.0 * ellipse_axes[0],
                height=2.0 * ellipse_axes[1],
                fill=False,
                edgecolor=color,
                linewidth=1.6,
                alpha=0.9,
                label=(
                    "effective chromosome boundary"
                    if chromosome_id == 0
                    else None
                ),
                zorder=2,
            )
            truth_axis.add_patch(ellipse)
            center_artist, = truth_axis.plot(
                [], [], marker="o", markersize=3.5, color=color, zorder=6
            )
            direction_artist, = truth_axis.plot(
                [], [], linewidth=1.2, color=color, alpha=0.95, zorder=3
            )
            trail_artist, = truth_axis.plot(
                [], [], linewidth=1.1, color=color, alpha=0.55, zorder=1
            )
            chromosome_ellipses.append(ellipse)
            chromosome_centers.append(center_artist)
            chromosome_directions.append(direction_artist)
            chromosome_trails.append(trail_artist)
        if protein_trail_data is not None:
            (
                protein_trail_particle_ids,
                protein_trail_chromosome_ids,
                protein_trail_frame_count,
            ) = protein_trail_data
            for trail_index, chromosome_id in enumerate(
                protein_trail_chromosome_ids
            ):
                protein_trail, = truth_axis.plot(
                    [],
                    [],
                    linewidth=1.0,
                    color=color_map(int(chromosome_id) % 10),
                    alpha=0.7,
                    label=(
                        "selected protein trails"
                        if trail_index == 0
                        else None
                    ),
                    zorder=3,
                )
                protein_trails.append(protein_trail)
        truth_axis.text(
            0.02,
            0.02,
            "Ellipses, 2 s center trails, and 1 s protein trails: private truth",
            transform=truth_axis.transAxes,
            color="#cbd5e1",
            fontsize=7.5,
            va="bottom",
            zorder=7,
        )

    truth_axis.legend(
        loc="upper right",
        facecolor="#111827",
        edgecolor="#38445a",
        labelcolor="white",
    )
    status_text = figure.suptitle("", color="white", fontsize=10)

    def update(frame: int):
        frame_positions = result.positions_um[frame]
        truth_positions = result.scenario.observation.project_positions(
            frame_positions
        )
        frame_emitting = result.emitting[frame]
        frame_active = result.active[frame]
        frame_in_region = result.in_observation_region[frame]
        frame_visible = result.visible[frame]
        emitting_in_region = frame_active & frame_in_region & frame_emitting
        non_emitting_in_region = frame_active & frame_in_region & ~frame_emitting
        outside_active = frame_active & ~frame_in_region
        inactive = ~frame_active

        truth_non_emitting.set_offsets(
            truth_positions[non_emitting_in_region]
            if np.any(non_emitting_in_region)
            else empty
        )
        truth_emitting.set_offsets(
            truth_positions[emitting_in_region]
            if np.any(emitting_in_region)
            else empty
        )
        observed_frame = result.observed_positions_um[frame]
        visible_positions = (
            observed_frame[frame_visible] if np.any(frame_visible) else empty
        )
        observed_glow.set_offsets(visible_positions)
        observed_points.set_offsets(visible_positions)

        if chromosome_data is not None:
            scene_centers, scene_orientations, _, ellipse_axes = chromosome_data
            trail_start = max(0, frame - trail_frame_count + 1)
            for chromosome_id, ellipse in enumerate(chromosome_ellipses):
                center = scene_centers[frame, chromosome_id]
                orientation = scene_orientations[frame, chromosome_id]
                ellipse.center = (float(center[0]), float(center[1]))
                ellipse.angle = float(np.degrees(orientation))
                chromosome_centers[chromosome_id].set_data(
                    [center[0]], [center[1]]
                )
                direction_end = center + 0.55 * ellipse_axes[0] * np.array(
                    [np.cos(orientation), np.sin(orientation)]
                )
                chromosome_directions[chromosome_id].set_data(
                    [center[0], direction_end[0]],
                    [center[1], direction_end[1]],
                )
                trail = scene_centers[
                    trail_start : frame + 1, chromosome_id
                ]
                chromosome_trails[chromosome_id].set_data(
                    trail[:, 0], trail[:, 1]
                )

            protein_trail_start = max(
                0, frame - protein_trail_frame_count + 1
            )
            for trail_artist, particle_id in zip(
                protein_trails, protein_trail_particle_ids, strict=True
            ):
                trail = result.positions_um[
                    protein_trail_start : frame + 1, particle_id
                ]
                projected_trail = result.scenario.observation.project_positions(
                    trail
                )
                trail_artist.set_data(
                    projected_trail[:, 0], projected_trail[:, 1]
                )

        status_text.set_text(
            f"{result.scenario.scenario_id}\n"
            f"t = {result.time_s[frame]:.2f} s   |   "
            f"in-FOV emitting {emitting_in_region.sum()}   "
            f"non-emitting {non_emitting_in_region.sum()}   "
            f"outside {outside_active.sum()}   "
            f"inactive {inactive.sum()}   visible {frame_visible.sum()}"
        )
        return (
            truth_non_emitting,
            truth_emitting,
            observed_glow,
            observed_points,
            *chromosome_ellipses,
            *chromosome_centers,
            *chromosome_directions,
            *chromosome_trails,
            *protein_trails,
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
