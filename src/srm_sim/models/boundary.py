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

    def apply_step(
        self,
        previous_positions_um: NDArray[np.float64],
        proposed_positions_um: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del previous_positions_um
        return self.apply(proposed_positions_um)

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

    def apply_step(
        self,
        previous_positions_um: NDArray[np.float64],
        proposed_positions_um: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del previous_positions_um
        return self.apply(proposed_positions_um)

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

    def apply_step(
        self,
        previous_positions_um: NDArray[np.float64],
        proposed_positions_um: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        del previous_positions_um
        return self.apply(proposed_positions_um)

    def parameters(self) -> Mapping[str, Any]:
        return {
            "physical_boundary": "none",
            "initialization_lengths_um": list(self.initialization_lengths_um),
        }


@dataclass(frozen=True)
class ReflectingEllipseBoundary:
    """Effective chromosome-local ellipse with specular no-flux reflection."""

    semi_major_axis_um: float = 2.0
    semi_minor_axis_um: float = 0.35
    max_reflections_per_step: int = 1024

    model_id: ClassVar[str] = "reflecting_ellipse_boundary_2d"
    dimension: ClassVar[int] = 2

    def __post_init__(self) -> None:
        axes = np.array(
            [self.semi_major_axis_um, self.semi_minor_axis_um],
            dtype=np.float64,
        )
        if not np.isfinite(axes).all() or np.any(axes <= 0.0):
            raise ValueError("ellipse semi-axes must be finite and positive")
        if (
            not isinstance(self.max_reflections_per_step, int)
            or isinstance(self.max_reflections_per_step, bool)
            or self.max_reflections_per_step <= 0
        ):
            raise ValueError("max_reflections_per_step must be a positive integer")

    @property
    def bounds_um(self) -> NDArray[np.float64]:
        return np.array(
            [
                [-self.semi_major_axis_um, self.semi_major_axis_um],
                [-self.semi_minor_axis_um, self.semi_minor_axis_um],
            ],
            dtype=np.float64,
        )

    def sample_initial_positions(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.float64]:
        radius = np.sqrt(rng.random(n_particles))
        angle = 2.0 * np.pi * rng.random(n_particles)
        return np.column_stack(
            (
                self.semi_major_axis_um * radius * np.cos(angle),
                self.semi_minor_axis_um * radius * np.sin(angle),
            )
        )

    def _scaled_radius_squared(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        axes = np.array(
            [self.semi_major_axis_um, self.semi_minor_axis_um],
            dtype=np.float64,
        )
        return np.sum(np.square(positions_um / axes), axis=-1)

    def apply(
        self, positions_um: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        positions = np.asarray(positions_um, dtype=np.float64)
        if positions.ndim != 2 or positions.shape[1] != self.dimension:
            raise ValueError("positions_um must have shape (particles, 2)")
        if np.any(self._scaled_radius_squared(positions) > 1.0 + 1e-12):
            raise ValueError(
                "ellipse reflection requires the previous position for an "
                "outside proposal"
            )
        return positions.copy()

    def apply_step(
        self,
        previous_positions_um: NDArray[np.float64],
        proposed_positions_um: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        previous = np.asarray(previous_positions_um, dtype=np.float64)
        proposed = np.asarray(proposed_positions_um, dtype=np.float64)
        if previous.shape != proposed.shape:
            raise ValueError("previous and proposed positions must have equal shapes")
        if previous.ndim != 2 or previous.shape[1] != self.dimension:
            raise ValueError("positions must have shape (particles, 2)")
        if not np.isfinite(previous).all() or not np.isfinite(proposed).all():
            raise ValueError("ellipse positions must be finite")
        tolerance = 1e-12
        previous_radius = self._scaled_radius_squared(previous)
        if np.any(previous_radius > 1.0 + tolerance):
            raise ValueError("previous positions must be inside the ellipse")

        canonical_previous = previous.copy()
        near_outside = previous_radius > 1.0
        canonical_previous[near_outside] /= np.sqrt(
            previous_radius[near_outside, None]
        )
        reflected = proposed.copy()
        crossing_indices = np.flatnonzero(
            self._scaled_radius_squared(proposed) > 1.0
        )
        for index in crossing_indices:
            reflected[index] = self._reflect_one(
                canonical_previous[index], proposed[index]
            )
        return reflected

    def _reflect_one(
        self,
        start_um: NDArray[np.float64],
        target_um: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        axes = np.array(
            [self.semi_major_axis_um, self.semi_minor_axis_um],
            dtype=np.float64,
        )
        axes_squared = axes * axes
        point = start_um.copy()
        displacement = target_um - start_um
        if not np.any(displacement):
            return point
        inward_nudge = 1e-12 * min(axes)

        for _ in range(self.max_reflections_per_step):
            target = point + displacement
            scaled_radius = float(self._scaled_radius_squared(target))
            if scaled_radius <= 1.0 + 1e-12:
                if scaled_radius > 1.0:
                    target /= np.sqrt(scaled_radius)
                return target

            quadratic_a = float(
                np.sum(displacement * displacement / axes_squared)
            )
            quadratic_b = float(
                2.0 * np.sum(point * displacement / axes_squared)
            )
            quadratic_c = float(np.sum(point * point / axes_squared) - 1.0)
            discriminant = max(
                0.0,
                quadratic_b * quadratic_b
                - 4.0 * quadratic_a * quadratic_c,
            )
            root_scale = 2.0 * quadratic_a
            roots = (
                (-quadratic_b - np.sqrt(discriminant)) / root_scale,
                (-quadratic_b + np.sqrt(discriminant)) / root_scale,
            )
            valid_roots = [
                root for root in roots if 1e-14 < root <= 1.0 + 1e-12
            ]
            if not valid_roots:
                valid_roots = [
                    root for root in roots if -1e-12 <= root <= 1.0 + 1e-12
                ]
            if not valid_roots:
                raise RuntimeError(
                    "could not find an ellipse boundary intersection"
                )
            fraction = min(valid_roots)
            hit = point + fraction * displacement
            normal = hit / axes_squared
            normal /= np.linalg.norm(normal)
            residual = (1.0 - fraction) * displacement
            normal_component = float(np.dot(residual, normal))
            if (
                fraction <= 1e-12
                and abs(normal_component)
                <= 1e-12 * np.linalg.norm(residual)
            ):
                projected = target / np.sqrt(scaled_radius)
                return projected * (1.0 - 1e-12)
            displacement = residual - 2.0 * normal_component * normal
            point = hit - inward_nudge * normal

        raise RuntimeError(
            "ellipse reflection exceeded max_reflections_per_step; "
            "reduce the time step or increase the reflection limit"
        )

    def parameters(self) -> Mapping[str, Any]:
        return {
            "semi_major_axis_um": self.semi_major_axis_um,
            "semi_minor_axis_um": self.semi_minor_axis_um,
            "coordinate_frame": "chromosome_local",
            "boundary_condition": (
                "finite_step_specular_reflection_with_grazing_projection"
            ),
            "max_reflections_per_step": self.max_reflections_per_step,
            "geometry_interpretation": (
                "effective_computational_confinement_mask"
            ),
        }
