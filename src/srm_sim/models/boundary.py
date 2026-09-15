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
