from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .scenario import BoundaryModel, SimulationScenario


@dataclass(frozen=True)
class SimulationConfig:
    n_particles: int = 100
    n_frames: int = 200
    frame_interval_s: float = 0.05
    random_seed: int = 7

    def __post_init__(self) -> None:
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive")
        if self.n_frames < 2:
            raise ValueError("n_frames must be at least 2")
        if self.frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be positive")


@dataclass(frozen=True)
class SimulationResult:
    config: SimulationConfig
    scenario: SimulationScenario
    time_s: NDArray[np.float64]
    particle_ids: NDArray[np.int64]
    bounds_um: NDArray[np.float64]
    positions_um: NDArray[np.float64]
    states: NDArray[np.uint8]
    emitting: NDArray[np.bool_]
    active: NDArray[np.bool_]
    in_observation_region: NDArray[np.bool_]
    observed_positions_um: NDArray[np.float64]
    visible: NDArray[np.bool_]
    local_positions_um: NDArray[np.float64] | None = None
    scene_centers_um: NDArray[np.float64] | None = None
    scene_orientations_rad: NDArray[np.float64] | None = None
    particle_scene_ids: NDArray[np.int64] | None = None
    scene_instance_centers_um: NDArray[np.float64] | None = None
    scene_instance_orientations_rad: NDArray[np.float64] | None = None


def _apply_boundary_step(
    boundary: BoundaryModel,
    previous_positions_um: NDArray[np.float64],
    proposed_positions_um: NDArray[np.float64],
) -> NDArray[np.float64]:
    apply_step = getattr(boundary, "apply_step", None)
    if apply_step is None:
        return boundary.apply(proposed_positions_um)
    return apply_step(previous_positions_um, proposed_positions_um)


def simulate(
    config: SimulationConfig, scenario: SimulationScenario
) -> SimulationResult:
    """Run a scenario while keeping truth and observations separate."""
    if scenario.multi_scene is not None:
        return _simulate_with_multi_scene(config, scenario)
    if scenario.scene is not None:
        return _simulate_with_scene(config, scenario)

    seed_sequence = np.random.SeedSequence(config.random_seed)
    boundary_seed, motion_seed, photophysics_seed, observation_seed = (
        seed_sequence.spawn(4)
    )
    boundary_rng = np.random.default_rng(boundary_seed)
    motion_rng = np.random.default_rng(motion_seed)
    photophysics_rng = np.random.default_rng(photophysics_seed)
    observation_rng = np.random.default_rng(observation_seed)

    shape = (config.n_frames, config.n_particles)
    physical_position_shape = (*shape, scenario.motion.dimension)
    observed_position_shape = (*shape, scenario.observation.output_dimension)

    positions = np.empty(physical_position_shape, dtype=np.float64)
    states = np.empty(shape, dtype=np.uint8)
    emitting = np.empty(shape, dtype=np.bool_)
    active = np.empty(shape, dtype=np.bool_)
    in_observation_region = np.empty(shape, dtype=np.bool_)
    observed_positions = np.empty(observed_position_shape, dtype=np.float64)
    visible = np.empty(shape, dtype=np.bool_)

    positions[0] = scenario.boundary.sample_initial_positions(
        config.n_particles, boundary_rng
    )
    states[0] = scenario.photophysics.initialize_states(
        config.n_particles, photophysics_rng
    )
    emitting[0] = scenario.photophysics.emitting_mask(states[0])
    active[0] = scenario.photophysics.active_mask(states[0])
    in_observation_region[0] = scenario.observation.in_observation_region(
        positions[0]
    )
    observed_positions[0], visible[0] = scenario.observation.observe(
        positions[0], emitting[0], observation_rng
    )

    for frame in range(1, config.n_frames):
        moved_positions = scenario.motion.step(
            positions[frame - 1],
            config.frame_interval_s,
            motion_rng,
        )
        positions[frame] = _apply_boundary_step(
            scenario.boundary,
            positions[frame - 1],
            moved_positions,
        )
        states[frame] = scenario.photophysics.step(
            states[frame - 1],
            config.frame_interval_s,
            photophysics_rng,
        )
        emitting[frame] = scenario.photophysics.emitting_mask(states[frame])
        active[frame] = scenario.photophysics.active_mask(states[frame])
        in_observation_region[frame] = (
            scenario.observation.in_observation_region(positions[frame])
        )
        observed_positions[frame], visible[frame] = scenario.observation.observe(
            positions[frame], emitting[frame], observation_rng
        )

    return SimulationResult(
        config=config,
        scenario=scenario,
        time_s=np.arange(config.n_frames, dtype=np.float64)
        * config.frame_interval_s,
        particle_ids=np.arange(config.n_particles, dtype=np.int64),
        bounds_um=scenario.observation.bounds_um.copy(),
        positions_um=positions,
        states=states,
        emitting=emitting,
        active=active,
        in_observation_region=in_observation_region,
        observed_positions_um=observed_positions,
        visible=visible,
    )


def _simulate_with_scene(
    config: SimulationConfig, scenario: SimulationScenario
) -> SimulationResult:
    scene = scenario.scene
    if scene is None:
        raise ValueError("scene-aware simulation requires a scene model")

    seed_sequence = np.random.SeedSequence(config.random_seed)
    (
        boundary_seed,
        motion_seed,
        photophysics_seed,
        observation_seed,
        scene_seed,
    ) = seed_sequence.spawn(5)
    boundary_rng = np.random.default_rng(boundary_seed)
    motion_rng = np.random.default_rng(motion_seed)
    photophysics_rng = np.random.default_rng(photophysics_seed)
    observation_rng = np.random.default_rng(observation_seed)
    scene_rng = np.random.default_rng(scene_seed)

    shape = (config.n_frames, config.n_particles)
    local_positions = np.empty((*shape, scene.local_dimension), dtype=np.float64)
    positions = np.empty((*shape, scene.lab_dimension), dtype=np.float64)
    scene_centers = np.empty((config.n_frames, scene.lab_dimension), dtype=np.float64)
    scene_orientations = np.empty(config.n_frames, dtype=np.float64)
    states = np.empty(shape, dtype=np.uint8)
    emitting = np.empty(shape, dtype=np.bool_)
    active = np.empty(shape, dtype=np.bool_)
    in_observation_region = np.empty(shape, dtype=np.bool_)
    observed_positions = np.empty(
        (*shape, scenario.observation.output_dimension), dtype=np.float64
    )
    visible = np.empty(shape, dtype=np.bool_)

    local_positions[0] = scenario.boundary.sample_initial_positions(
        config.n_particles, boundary_rng
    )
    pose = scene.initialize_pose(scene_rng)
    scene_centers[0] = pose.center_um
    scene_orientations[0] = pose.orientation_rad
    positions[0] = scene.transform(local_positions[0], pose)
    states[0] = scenario.photophysics.initialize_states(
        config.n_particles, photophysics_rng
    )
    emitting[0] = scenario.photophysics.emitting_mask(states[0])
    active[0] = scenario.photophysics.active_mask(states[0])
    in_observation_region[0] = scenario.observation.in_observation_region(
        positions[0]
    )
    observed_positions[0], visible[0] = scenario.observation.observe(
        positions[0], emitting[0], observation_rng
    )

    for frame in range(1, config.n_frames):
        proposed_local_positions = scenario.motion.step(
            local_positions[frame - 1],
            config.frame_interval_s,
            motion_rng,
        )
        local_positions[frame] = _apply_boundary_step(
            scenario.boundary,
            local_positions[frame - 1],
            proposed_local_positions,
        )
        pose = scene.step_pose(pose, config.frame_interval_s, scene_rng)
        scene_centers[frame] = pose.center_um
        scene_orientations[frame] = pose.orientation_rad
        positions[frame] = scene.transform(local_positions[frame], pose)
        states[frame] = scenario.photophysics.step(
            states[frame - 1],
            config.frame_interval_s,
            photophysics_rng,
        )
        emitting[frame] = scenario.photophysics.emitting_mask(states[frame])
        active[frame] = scenario.photophysics.active_mask(states[frame])
        in_observation_region[frame] = (
            scenario.observation.in_observation_region(positions[frame])
        )
        observed_positions[frame], visible[frame] = scenario.observation.observe(
            positions[frame], emitting[frame], observation_rng
        )

    return SimulationResult(
        config=config,
        scenario=scenario,
        time_s=np.arange(config.n_frames, dtype=np.float64)
        * config.frame_interval_s,
        particle_ids=np.arange(config.n_particles, dtype=np.int64),
        bounds_um=scenario.observation.bounds_um.copy(),
        positions_um=positions,
        states=states,
        emitting=emitting,
        active=active,
        in_observation_region=in_observation_region,
        observed_positions_um=observed_positions,
        visible=visible,
        local_positions_um=local_positions,
        scene_centers_um=scene_centers,
        scene_orientations_rad=scene_orientations,
    )


def _simulate_with_multi_scene(
    config: SimulationConfig, scenario: SimulationScenario
) -> SimulationResult:
    scene = scenario.multi_scene
    if scene is None:
        raise ValueError("multi-scene simulation requires a multi-scene model")
    chromosome_count = scene.instance_count
    if config.n_particles < chromosome_count:
        raise ValueError("n_particles must be at least the chromosome count")

    seed_sequence = np.random.SeedSequence(config.random_seed)
    (
        boundary_seed,
        motion_seed,
        photophysics_seed,
        observation_seed,
        scene_seed,
    ) = seed_sequence.spawn(5)
    boundary_rngs = tuple(
        np.random.default_rng(seed)
        for seed in boundary_seed.spawn(chromosome_count)
    )
    motion_rngs = tuple(
        np.random.default_rng(seed)
        for seed in motion_seed.spawn(chromosome_count)
    )
    scene_seeds = scene_seed.spawn(chromosome_count + 1)
    scene_initialization_rng = np.random.default_rng(scene_seeds[0])
    scene_motion_rngs = tuple(
        np.random.default_rng(seed) for seed in scene_seeds[1:]
    )
    photophysics_rng = np.random.default_rng(photophysics_seed)
    observation_rng = np.random.default_rng(observation_seed)

    counts = np.full(
        chromosome_count,
        config.n_particles // chromosome_count,
        dtype=np.int64,
    )
    counts[: config.n_particles % chromosome_count] += 1
    particle_scene_ids = np.repeat(
        np.arange(chromosome_count, dtype=np.int64), counts
    )

    shape = (config.n_frames, config.n_particles)
    local_positions = np.empty(
        (*shape, scene.local_dimension), dtype=np.float64
    )
    positions = np.empty((*shape, scene.lab_dimension), dtype=np.float64)
    scene_centers = np.empty(
        (config.n_frames, chromosome_count, scene.lab_dimension),
        dtype=np.float64,
    )
    scene_orientations = np.empty(
        (config.n_frames, chromosome_count), dtype=np.float64
    )
    states = np.empty(shape, dtype=np.uint8)
    emitting = np.empty(shape, dtype=np.bool_)
    active = np.empty(shape, dtype=np.bool_)
    in_observation_region = np.empty(shape, dtype=np.bool_)
    observed_positions = np.empty(
        (*shape, scenario.observation.output_dimension), dtype=np.float64
    )
    visible = np.empty(shape, dtype=np.bool_)

    for chromosome_id, boundary_rng in enumerate(boundary_rngs):
        selected = particle_scene_ids == chromosome_id
        local_positions[0, selected] = (
            scenario.boundary.sample_initial_positions(
                int(selected.sum()), boundary_rng
            )
        )
    scene_state = scene.initialize_state(scene_initialization_rng)
    if len(scene_state.poses) != chromosome_count:
        raise ValueError("initialized pose count does not match instance_count")
    for chromosome_id, pose in enumerate(scene_state.poses):
        scene_centers[0, chromosome_id] = pose.center_um
        scene_orientations[0, chromosome_id] = pose.orientation_rad
    positions[0] = scene.transform(
        local_positions[0], particle_scene_ids, scene_state
    )
    states[0] = scenario.photophysics.initialize_states(
        config.n_particles, photophysics_rng
    )
    emitting[0] = scenario.photophysics.emitting_mask(states[0])
    active[0] = scenario.photophysics.active_mask(states[0])
    in_observation_region[0] = scenario.observation.in_observation_region(
        positions[0]
    )
    observed_positions[0], visible[0] = scenario.observation.observe(
        positions[0], emitting[0], observation_rng
    )

    for frame in range(1, config.n_frames):
        for chromosome_id, motion_rng in enumerate(motion_rngs):
            selected = particle_scene_ids == chromosome_id
            previous = local_positions[frame - 1, selected]
            proposed = scenario.motion.step(
                previous, config.frame_interval_s, motion_rng
            )
            local_positions[frame, selected] = _apply_boundary_step(
                scenario.boundary, previous, proposed
            )
        scene_state = scene.step_state(
            scene_state,
            config.frame_interval_s,
            scene_motion_rngs,
        )
        for chromosome_id, pose in enumerate(scene_state.poses):
            scene_centers[frame, chromosome_id] = pose.center_um
            scene_orientations[frame, chromosome_id] = pose.orientation_rad
        positions[frame] = scene.transform(
            local_positions[frame], particle_scene_ids, scene_state
        )
        states[frame] = scenario.photophysics.step(
            states[frame - 1],
            config.frame_interval_s,
            photophysics_rng,
        )
        emitting[frame] = scenario.photophysics.emitting_mask(states[frame])
        active[frame] = scenario.photophysics.active_mask(states[frame])
        in_observation_region[frame] = (
            scenario.observation.in_observation_region(positions[frame])
        )
        observed_positions[frame], visible[frame] = (
            scenario.observation.observe(
                positions[frame], emitting[frame], observation_rng
            )
        )

    return SimulationResult(
        config=config,
        scenario=scenario,
        time_s=np.arange(config.n_frames, dtype=np.float64)
        * config.frame_interval_s,
        particle_ids=np.arange(config.n_particles, dtype=np.int64),
        bounds_um=scenario.observation.bounds_um.copy(),
        positions_um=positions,
        states=states,
        emitting=emitting,
        active=active,
        in_observation_region=in_observation_region,
        observed_positions_um=observed_positions,
        visible=visible,
        local_positions_um=local_positions,
        particle_scene_ids=particle_scene_ids,
        scene_instance_centers_um=scene_centers,
        scene_instance_orientations_rad=scene_orientations,
    )
