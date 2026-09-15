from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ReflectingSquareBoundary:
    field_size_um: float = 10.0

    model_id: ClassVar[str] = "reflecting_square_boundary_2d"
    dimension: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if self.field_size_um <= 0:
            raise ValueError("field_size_um must be positive")

    @property
    def bounds_um(self) -> NDArray[np.float64]:
        return np.array(
            [[0.0, self.field_size_um], [0.0, self.field_size_um]],
            dtype=np.float64,
        )

    def sample_initial_positions(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return rng.uniform(
            0.0,
            self.field_size_um,
            size=(n_particles, self.dimension),
        )

    def apply(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        period = 2.0 * self.field_size_um
        return self.field_size_um - np.abs(
            np.mod(positions_um, period) - self.field_size_um
        )

    def parameters(self) -> Mapping[str, Any]:
        return {"field_size_um": self.field_size_um}


@dataclass(frozen=True)
class UnboundedPlane2D:
    """Free plane with a finite square region used only for initialization."""

    initialization_size_um: float = 30.0

    model_id: ClassVar[str] = "unbounded_plane_2d"
    dimension: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if self.initialization_size_um <= 0:
            raise ValueError("initialization_size_um must be positive")

    @property
    def bounds_um(self) -> NDArray[np.float64]:
        """Return initialization bounds, not physical motion boundaries."""
        return np.array(
            [
                [0.0, self.initialization_size_um],
                [0.0, self.initialization_size_um],
            ],
            dtype=np.float64,
        )

    def sample_initial_positions(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return rng.uniform(
            0.0,
            self.initialization_size_um,
            size=(n_particles, self.dimension),
        )

    def apply(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return positions_um.copy()

    def parameters(self) -> Mapping[str, Any]:
        return {
            "physical_boundary": "none",
            "initialization_size_um": self.initialization_size_um,
        }


@dataclass(frozen=True)
class UnboundedSpace3D:
    initialization_lengths_um: tuple[float, float, float] = (30.0, 30.0, 22.0)

    model_id: ClassVar[str] = "unbounded_space_3d"
    dimension: ClassVar[int] = 3

    def __post_init__(self) -> None:
        lengths = np.asarray(self.initialization_lengths_um, dtype=np.float64)
        if lengths.shape != (self.dimension,):
            raise ValueError("initialization lengths must contain three values")
        if not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("initialization lengths must be finite and positive")

    @property
    def bounds_um(self) -> NDArray[np.float64]:
        return np.column_stack(
            (
                np.zeros(self.dimension, dtype=np.float64),
                np.asarray(self.initialization_lengths_um, dtype=np.float64),
            )
        )

    def sample_initial_positions(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return rng.uniform(
            np.zeros(self.dimension, dtype=np.float64),
            np.asarray(self.initialization_lengths_um, dtype=np.float64),
            size=(n_particles, self.dimension),
        )

    def apply(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return positions_um.copy()

    def parameters(self) -> Mapping[str, Any]:
        return {
            "physical_boundary": "none",
            "initialization_lengths_um": list(self.initialization_lengths_um),
        }
