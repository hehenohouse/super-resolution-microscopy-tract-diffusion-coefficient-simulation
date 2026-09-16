from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RigidPose2D:
    center_um: tuple[float, float]
    orientation_rad: float


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


class StepBoundaryModel(BoundaryModel, Protocol):
    def apply_step(
        self,
        previous_positions_um: NDArray[np.float64],
        proposed_positions_um: NDArray[np.float64],
    ) -> NDArray[np.float64]: ...


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


class SceneModel(Protocol):
    model_id: str
    local_dimension: int
    lab_dimension: int

    def initialize_pose(self, rng: np.random.Generator) -> RigidPose2D: ...

    def step_pose(
        self,
        pose: RigidPose2D,
        frame_interval_s: float,
        rng: np.random.Generator,
    ) -> RigidPose2D: ...

    def transform(
        self,
        local_positions_um: NDArray[np.float64],
        pose: RigidPose2D,
    ) -> NDArray[np.float64]: ...

    def parameters(self) -> Mapping[str, Any]: ...


class MultiSceneModel(Protocol):
    model_id: str
    local_dimension: int
    lab_dimension: int
    instance_count: int

    def initialize_state(self, rng: np.random.Generator) -> Any: ...

    def step_state(
        self,
        state: Any,
        frame_interval_s: float,
        rngs: tuple[np.random.Generator, ...],
    ) -> Any: ...

    def transform(
        self,
        local_positions_um: NDArray[np.float64],
        instance_ids: NDArray[np.int64],
        state: Any,
    ) -> NDArray[np.float64]: ...

    def parameters(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SimulationScenario:
    scenario_id: str
    name: str
    motion: MotionModel
    boundary: BoundaryModel
    photophysics: PhotophysicsModel
    observation: ObservationModel
    scene: SceneModel | None = None
    multi_scene: MultiSceneModel | None = None

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id cannot be empty")
        if self.scene is not None and self.multi_scene is not None:
            raise ValueError("scene and multi_scene are mutually exclusive")
        active_scene = self.scene if self.scene is not None else self.multi_scene
        if active_scene is None:
            physical_dimensions = {
                self.motion.dimension,
                self.boundary.dimension,
                self.observation.input_dimension,
            }
            dimension_error = (
                "motion, boundary, and observation input dimensions must match"
            )
        else:
            if active_scene.local_dimension != 2 or active_scene.lab_dimension != 2:
                raise ValueError("RigidPose2D scene dimensions must both be two")
            if self.multi_scene is not None and self.multi_scene.instance_count <= 0:
                raise ValueError("multi_scene instance_count must be positive")
            physical_dimensions = {
                self.motion.dimension,
                self.boundary.dimension,
                active_scene.local_dimension,
            }
            dimension_error = (
                "motion, boundary, and scene local dimensions must match"
            )
            if active_scene.lab_dimension != self.observation.input_dimension:
                raise ValueError(
                    "scene lab and observation input dimensions must match"
                )
        if len(physical_dimensions) != 1:
            raise ValueError(dimension_error)
        if self.observation.output_dimension <= 0:
            raise ValueError("observation output dimension must be positive")
        if self.observation.bounds_um.shape != (
            self.observation.output_dimension,
            2,
        ):
            raise ValueError("observation bounds do not match its output dimension")

    def metadata(self) -> dict[str, Any]:
        components = {
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
        }
        active_scene = self.scene if self.scene is not None else self.multi_scene
        if active_scene is not None:
            components["scene"] = {
                "model_id": active_scene.model_id,
                "parameters": dict(active_scene.parameters()),
            }
        return {
            "scenario_id": self.scenario_id,
            "scenario_name": self.name,
            "physical_dimension": (
                self.motion.dimension
                if active_scene is None
                else active_scene.lab_dimension
            ),
            "observation_dimension": self.observation.output_dimension,
            "components": components,
        }
