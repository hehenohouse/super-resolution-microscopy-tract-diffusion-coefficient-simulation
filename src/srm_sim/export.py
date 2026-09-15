from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .simulator import SimulationResult


OBSERVATION_FORMAT_VERSION = "1.0"
GROUND_TRUTH_FORMAT_VERSION = "1.0"


@dataclass(frozen=True)
class ObservationTable:
    scenario_id: str
    observation_model: str
    n_frames: int
    frame_interval_s: float
    frame_times_s: NDArray[np.float64]
    bounds_um: NDArray[np.float64]
    detection_order_seed: int
    detection_ids: NDArray[np.int64]
    frame_indices: NDArray[np.int64]
    times_s: NDArray[np.float64]
    positions_um: NDArray[np.float64]

    def __post_init__(self) -> None:
        n_detections = self.detection_ids.size
        if self.n_frames <= 0:
            raise ValueError("n_frames must be positive")
        if self.frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be positive")
        if self.frame_times_s.shape != (self.n_frames,):
            raise ValueError("frame_times_s has an invalid shape")
        if self.frame_indices.shape != (n_detections,):
            raise ValueError("frame_indices has an invalid shape")
        if self.times_s.shape != (n_detections,):
            raise ValueError("times_s has an invalid shape")
        if self.positions_um.ndim != 2:
            raise ValueError("positions_um must have shape (detections, dimensions)")
        if self.positions_um.shape[0] != n_detections:
            raise ValueError("positions_um and detection_ids must have equal lengths")
        if self.bounds_um.shape != (self.positions_um.shape[1], 2):
            raise ValueError("bounds_um does not match the observation dimensions")
        if np.unique(self.detection_ids).size != n_detections:
            raise ValueError("detection_ids must be unique")
        if n_detections:
            if self.frame_indices.min() < 0 or self.frame_indices.max() >= self.n_frames:
                raise ValueError("frame_indices contains an out-of-range frame")
            if not np.isfinite(self.positions_um).all():
                raise ValueError("public detection coordinates must all be finite")
            expected_times = self.frame_times_s[self.frame_indices]
            if not np.array_equal(self.times_s, expected_times):
                raise ValueError("detection times do not match their frame indices")

    def for_frame(
        self, frame_index: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return detection IDs and the unordered point cloud for one frame."""
        if not 0 <= frame_index < self.n_frames:
            raise IndexError(f"frame_index must be in [0, {self.n_frames})")
        in_frame = self.frame_indices == frame_index
        return self.detection_ids[in_frame], self.positions_um[in_frame]


@dataclass(frozen=True)
class OutputPaths:
    observations: Path
    ground_truth: Path


@dataclass(frozen=True)
class _DetectionExport:
    observations: ObservationTable
    true_particle_ids: NDArray[np.int64]
    true_positions_um: NDArray[np.float64]


def _build_detection_export(
    result: SimulationResult, detection_order_seed: int
) -> _DetectionExport:
    rng = np.random.default_rng(detection_order_seed)
    dimension = result.positions_um.shape[2]

    frame_parts: list[NDArray[np.int64]] = []
    time_parts: list[NDArray[np.float64]] = []
    observed_position_parts: list[NDArray[np.float64]] = []
    true_particle_parts: list[NDArray[np.int64]] = []
    true_position_parts: list[NDArray[np.float64]] = []

    for frame_index in range(result.config.n_frames):
        particle_slots = np.flatnonzero(result.visible[frame_index])
        rng.shuffle(particle_slots)
        count = particle_slots.size
        if count == 0:
            continue

        frame_parts.append(np.full(count, frame_index, dtype=np.int64))
        time_parts.append(
            np.full(count, result.time_s[frame_index], dtype=np.float64)
        )
        observed_position_parts.append(
            result.observed_positions_um[frame_index, particle_slots].copy()
        )
        true_particle_parts.append(result.particle_ids[particle_slots].copy())
        true_position_parts.append(
            result.positions_um[frame_index, particle_slots].copy()
        )

    if frame_parts:
        frame_indices = np.concatenate(frame_parts)
        times_s = np.concatenate(time_parts)
        observed_positions_um = np.concatenate(observed_position_parts)
        true_particle_ids = np.concatenate(true_particle_parts)
        true_positions_um = np.concatenate(true_position_parts)
    else:
        frame_indices = np.empty(0, dtype=np.int64)
        times_s = np.empty(0, dtype=np.float64)
        observed_positions_um = np.empty((0, dimension), dtype=np.float64)
        true_particle_ids = np.empty(0, dtype=np.int64)
        true_positions_um = np.empty((0, dimension), dtype=np.float64)

    detection_ids = np.arange(frame_indices.size, dtype=np.int64)
    observation_model = result.scenario.observation.model_id
    observations = ObservationTable(
        scenario_id=result.scenario.scenario_id,
        observation_model=observation_model,
        n_frames=result.config.n_frames,
        frame_interval_s=result.config.frame_interval_s,
        frame_times_s=result.time_s.copy(),
        bounds_um=result.bounds_um.copy(),
        detection_order_seed=detection_order_seed,
        detection_ids=detection_ids,
        frame_indices=frame_indices,
        times_s=times_s,
        positions_um=observed_positions_um,
    )
    return _DetectionExport(
        observations=observations,
        true_particle_ids=true_particle_ids,
        true_positions_um=true_positions_um,
    )


def save_output_bundle(
    result: SimulationResult,
    output_directory: str | Path,
    *,
    detection_order_seed: int = 17,
) -> OutputPaths:
    """Write tracking-safe observations and private ground truth separately."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = OutputPaths(
        observations=output_directory / "observations.npz",
        ground_truth=output_directory / "ground_truth.npz",
    )

    exported = _build_detection_export(result, detection_order_seed)
    observations = exported.observations
    np.savez_compressed(
        paths.observations,
        format_version=OBSERVATION_FORMAT_VERSION,
        scenario_id=observations.scenario_id,
        observation_model=observations.observation_model,
        position_unit="um",
        time_unit="s",
        n_frames=observations.n_frames,
        frame_interval_s=observations.frame_interval_s,
        frame_times_s=observations.frame_times_s,
        bounds_um=observations.bounds_um,
        detection_order_seed=observations.detection_order_seed,
        detection_ids=observations.detection_ids,
        frame_indices=observations.frame_indices,
        times_s=observations.times_s,
        positions_um=observations.positions_um,
    )

    scenario_metadata = result.scenario.metadata()
    components = scenario_metadata["components"]
    np.savez_compressed(
        paths.ground_truth,
        format_version=GROUND_TRUTH_FORMAT_VERSION,
        scenario_id=result.scenario.scenario_id,
        scenario_name=result.scenario.name,
        motion_model=components["motion"]["model_id"],
        boundary_model=components["boundary"]["model_id"],
        photophysics_model=components["photophysics"]["model_id"],
        observation_model=components["observation"]["model_id"],
        scenario_json=json.dumps(scenario_metadata, sort_keys=True),
        config_json=json.dumps(asdict(result.config), sort_keys=True),
        frame_times_s=result.time_s,
        particle_ids=result.particle_ids,
        bounds_um=result.bounds_um,
        positions_um=result.positions_um,
        states=result.states,
        emitting=result.emitting,
        active=result.active,
        detection_ids=observations.detection_ids,
        detection_frame_indices=observations.frame_indices,
        detection_particle_ids=exported.true_particle_ids,
        detection_true_positions_um=exported.true_positions_um,
    )
    return paths


def load_observations(path: str | Path) -> ObservationTable:
    """Load only the public detection table used by a tracking method."""
    with np.load(Path(path), allow_pickle=False) as data:
        format_version = str(data["format_version"])
        if format_version != OBSERVATION_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported observation format version: {format_version}"
            )
        return ObservationTable(
            scenario_id=str(data["scenario_id"]),
            observation_model=str(data["observation_model"]),
            n_frames=int(data["n_frames"]),
            frame_interval_s=float(data["frame_interval_s"]),
            frame_times_s=data["frame_times_s"].copy(),
            bounds_um=data["bounds_um"].copy(),
            detection_order_seed=int(data["detection_order_seed"]),
            detection_ids=data["detection_ids"].copy(),
            frame_indices=data["frame_indices"].copy(),
            times_s=data["times_s"].copy(),
            positions_um=data["positions_um"].copy(),
        )
