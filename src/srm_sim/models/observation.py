from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IdealOnStateObservation:
    lower_bounds_um: tuple[float, float] = (0.0, 0.0)
    upper_bounds_um: tuple[float, float] = (10.0, 10.0)

    model_id: ClassVar[str] = "ideal_emitter_localization_in_rectangular_fov"
    input_dimension: ClassVar[int] = 2
    output_dimension: ClassVar[int] = 2

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower_bounds_um, dtype=np.float64)
        upper = np.asarray(self.upper_bounds_um, dtype=np.float64)
        if lower.shape != (self.input_dimension,) or upper.shape != (
            self.input_dimension,
        ):
            raise ValueError("observation bounds must contain one value per dimension")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("observation bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("each upper observation bound must exceed its lower bound")

    @property
    def bounds_um(self) -> NDArray[np.float64]:
        return np.column_stack((self.lower_bounds_um, self.upper_bounds_um)).astype(
            np.float64
        )

    def in_observation_region(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        if positions_um.ndim != 2 or positions_um.shape[1] != self.input_dimension:
            raise ValueError("positions_um has an invalid shape")
        lower = np.asarray(self.lower_bounds_um, dtype=np.float64)
        upper = np.asarray(self.upper_bounds_um, dtype=np.float64)
        return np.all((positions_um >= lower) & (positions_um <= upper), axis=1)

    def project_positions(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if positions_um.ndim != 2 or positions_um.shape[1] != self.input_dimension:
            raise ValueError("positions_um has an invalid shape")
        return positions_um.copy()

    def observe(
        self,
        positions_um: NDArray[np.float64],
        emitting: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        del rng
        visible = emitting & self.in_observation_region(positions_um)
        projected_positions_um = self.project_positions(positions_um)
        observed_positions_um = np.full_like(projected_positions_um, np.nan)
        observed_positions_um[visible] = projected_positions_um[visible]
        return observed_positions_um, visible

    def parameters(self) -> Mapping[str, Any]:
        return {
            "lower_bounds_um": list(self.lower_bounds_um),
            "upper_bounds_um": list(self.upper_bounds_um),
        }


@dataclass(frozen=True)
class IdealAxialSlabProjectionObservation:
    lower_bounds_um: tuple[float, float, float] = (10.0, 10.0, 10.0)
    upper_bounds_um: tuple[float, float, float] = (20.0, 20.0, 12.0)

    model_id: ClassVar[str] = "ideal_axial_slab_xy_projection"
    input_dimension: ClassVar[int] = 3
    output_dimension: ClassVar[int] = 2

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower_bounds_um, dtype=np.float64)
        upper = np.asarray(self.upper_bounds_um, dtype=np.float64)
        if lower.shape != (self.input_dimension,) or upper.shape != (
            self.input_dimension,
        ):
            raise ValueError("observation bounds must contain x, y, and z values")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("observation bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("each upper observation bound must exceed its lower bound")

    @property
    def bounds_um(self) -> NDArray[np.float64]:
        return np.column_stack(
            (self.lower_bounds_um[:2], self.upper_bounds_um[:2])
        ).astype(np.float64)

    def in_observation_region(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.bool_]:
        if positions_um.ndim != 2 or positions_um.shape[1] != self.input_dimension:
            raise ValueError("positions_um has an invalid shape")
        lower = np.asarray(self.lower_bounds_um, dtype=np.float64)
        upper = np.asarray(self.upper_bounds_um, dtype=np.float64)
        return np.all((positions_um >= lower) & (positions_um <= upper), axis=1)

    def project_positions(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if positions_um.ndim != 2 or positions_um.shape[1] != self.input_dimension:
            raise ValueError("positions_um has an invalid shape")
        return positions_um[:, : self.output_dimension].copy()

    def observe(
        self,
        positions_um: NDArray[np.float64],
        emitting: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        del rng
        visible = emitting & self.in_observation_region(positions_um)
        projected_positions_um = self.project_positions(positions_um)
        observed_positions_um = np.full_like(projected_positions_um, np.nan)
        observed_positions_um[visible] = projected_positions_um[visible]
        return observed_positions_um, visible

    def parameters(self) -> Mapping[str, Any]:
        return {
            "lower_bounds_um": list(self.lower_bounds_um),
            "upper_bounds_um": list(self.upper_bounds_um),
            "projection": "xy",
        }
