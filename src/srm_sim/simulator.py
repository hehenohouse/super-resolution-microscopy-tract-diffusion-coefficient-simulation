from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .scenario import SimulationScenario


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


def simulate(
    config: SimulationConfig, scenario: SimulationScenario
) -> SimulationResult:
    """Run a scenario while keeping truth and observations separate."""
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
        positions[frame] = scenario.boundary.apply(moved_positions)
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
