from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray


class MotionModel(Protocol):
    model_id: str
    dimension: int

    def step(
        self,
        positions_um: NDArray[np.float64],
        frame_interval_s: float,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]: ...

    def parameters(self) -> Mapping[str, Any]: ...


class BoundaryModel(Protocol):
    model_id: str
    dimension: int

    @property
    def bounds_um(self) -> NDArray[np.float64]: ...

    def sample_initial_positions(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]: ...

    def apply(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def parameters(self) -> Mapping[str, Any]: ...


class PhotophysicsModel(Protocol):
    model_id: str

    def initialize_states(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.uint8]: ...

    def step(
        self,
        states: NDArray[np.uint8],
        frame_interval_s: float,
        rng: np.random.Generator,
    ) -> NDArray[np.uint8]: ...

    def emitting_mask(
        self, states: NDArray[np.uint8]
    ) -> NDArray[np.bool_]: ...

    def active_mask(
        self, states: NDArray[np.uint8]
    ) -> NDArray[np.bool_]: ...

    def parameters(self) -> Mapping[str, Any]: ...


class ObservationModel(Protocol):
    model_id: str
    input_dimension: int
    output_dimension: int

    @property
    def bounds_um(self) -> NDArray[np.float64]: ...

    def in_observation_region(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.bool_]: ...

    def project_positions(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def observe(
        self,
        positions_um: NDArray[np.float64],
        emitting: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]: ...

    def parameters(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SimulationScenario:
    scenario_id: str
    name: str
    motion: MotionModel
    boundary: BoundaryModel
    photophysics: PhotophysicsModel
    observation: ObservationModel

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id cannot be empty")
        physical_dimensions = {
            self.motion.dimension,
            self.boundary.dimension,
            self.observation.input_dimension,
        }
        if len(physical_dimensions) != 1:
            raise ValueError(
                "motion, boundary, and observation input dimensions must match"
            )
        if self.observation.output_dimension <= 0:
            raise ValueError("observation output dimension must be positive")
        if self.observation.bounds_um.shape != (
            self.observation.output_dimension,
            2,
        ):
            raise ValueError("observation bounds do not match its output dimension")

    def metadata(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.name,
            "physical_dimension": self.motion.dimension,
            "observation_dimension": self.observation.output_dimension,
            "components": {
                "motion": {
                    "model_id": self.motion.model_id,
                    "parameters": dict(self.motion.parameters()),
                },
                "boundary": {
                    "model_id": self.boundary.model_id,
                    "parameters": dict(self.boundary.parameters()),
                },
                "photophysics": {
                    "model_id": self.photophysics.model_id,
                    "parameters": dict(self.photophysics.parameters()),
                },
                "observation": {
                    "model_id": self.observation.model_id,
                    "parameters": dict(self.observation.parameters()),
                },
            },
        }
