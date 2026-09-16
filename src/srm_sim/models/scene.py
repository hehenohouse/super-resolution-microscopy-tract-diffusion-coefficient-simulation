from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np
from numpy.typing import NDArray

from srm_sim.scenario import RigidPose2D


@dataclass(frozen=True)
class DynamicChromosomeScene2D:
    chromosome_diffusion_coefficient_um2_s: float = 0.001
    chromosome_relaxation_rate_s: float = 1.0 / 30.0
    rotational_diffusion_rad2_s: float = 0.001
    equilibrium_center_um: tuple[float, float] = (5.0, 5.0)
    initial_center_um: tuple[float, float] = (5.0, 5.0)
    initial_orientation_rad: float = 0.0

    model_id: ClassVar[str] = "dynamic_chromosome_rigid_pose_2d"
    local_dimension: ClassVar[int] = 2
    lab_dimension: ClassVar[int] = 2

    def __post_init__(self) -> None:
        rates = np.array(
            [
                self.chromosome_diffusion_coefficient_um2_s,
                self.chromosome_relaxation_rate_s,
                self.rotational_diffusion_rad2_s,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(rates).all() or np.any(rates < 0.0):
            raise ValueError(
                "chromosome diffusion and relaxation rates must be finite "
                "and nonnegative"
            )
        equilibrium = np.asarray(self.equilibrium_center_um, dtype=np.float64)
        initial = np.asarray(self.initial_center_um, dtype=np.float64)
        if equilibrium.shape != (2,) or initial.shape != (2,):
            raise ValueError("chromosome centers must contain x and y")
        if not np.isfinite(equilibrium).all() or not np.isfinite(initial).all():
            raise ValueError("chromosome centers must be finite")
        if not np.isfinite(self.initial_orientation_rad):
            raise ValueError("initial_orientation_rad must be finite")

    def initialize_pose(self, rng: np.random.Generator) -> RigidPose2D:
        del rng
        return RigidPose2D(
            center_um=(
                float(self.initial_center_um[0]),
                float(self.initial_center_um[1]),
            ),
            orientation_rad=float(self.initial_orientation_rad),
        )

    def step_pose(
        self,
        pose: RigidPose2D,
        frame_interval_s: float,
        rng: np.random.Generator,
    ) -> RigidPose2D:
        if frame_interval_s <= 0.0:
            raise ValueError("frame_interval_s must be positive")

        center_innovation = rng.normal(0.0, 1.0, size=2)
        orientation_innovation = float(rng.normal(0.0, 1.0))
        center = np.asarray(pose.center_um, dtype=np.float64)
        equilibrium = np.asarray(self.equilibrium_center_um, dtype=np.float64)
        diffusion = self.chromosome_diffusion_coefficient_um2_s
        relaxation = self.chromosome_relaxation_rate_s
        if diffusion == 0.0:
            next_center = center.copy()
        elif relaxation == 0.0:
            next_center = center + (
                np.sqrt(2.0 * diffusion * frame_interval_s)
                * center_innovation
            )
        else:
            decay = np.exp(-relaxation * frame_interval_s)
            noise_std = np.sqrt(
                diffusion
                / relaxation
                * -np.expm1(-2.0 * relaxation * frame_interval_s)
            )
            next_center = (
                equilibrium
                + decay * (center - equilibrium)
                + noise_std * center_innovation
            )

        rotational_diffusion = self.rotational_diffusion_rad2_s
        if rotational_diffusion == 0.0:
            next_orientation = pose.orientation_rad
        else:
            next_orientation = pose.orientation_rad + (
                np.sqrt(2.0 * rotational_diffusion * frame_interval_s)
                * orientation_innovation
            )
        return RigidPose2D(
            center_um=(float(next_center[0]), float(next_center[1])),
            orientation_rad=float(next_orientation),
        )

    def transform(
        self,
        local_positions_um: NDArray[np.float64],
        pose: RigidPose2D,
    ) -> NDArray[np.float64]:
        if local_positions_um.ndim != 2 or local_positions_um.shape[1] != 2:
            raise ValueError("local_positions_um must have shape (particles, 2)")
        cosine = np.cos(pose.orientation_rad)
        sine = np.sin(pose.orientation_rad)
        rotation = np.array(
            [[cosine, -sine], [sine, cosine]], dtype=np.float64
        )
        center = np.asarray(pose.center_um, dtype=np.float64)
        return local_positions_um @ rotation.T + center

    def parameters(self) -> Mapping[str, Any]:
        return {
            "chromosome_diffusion_coefficient_um2_s": (
                self.chromosome_diffusion_coefficient_um2_s
            ),
            "chromosome_relaxation_rate_s": self.chromosome_relaxation_rate_s,
            "rotational_diffusion_rad2_s": self.rotational_diffusion_rad2_s,
            "equilibrium_center_um": list(self.equilibrium_center_um),
            "initial_center_um": list(self.initial_center_um),
            "initial_orientation_rad": self.initial_orientation_rad,
        }


@dataclass(frozen=True)
class MultiChromosomeState2D:
    poses: tuple[RigidPose2D, ...]
    equilibrium_centers_um: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class IndependentChromosomesScene2D:
    instance_count: int = 6
    chromosome_diffusion_coefficient_um2_s: float = 0.001
    chromosome_relaxation_rate_s: float = 1.0 / 30.0
    rotational_diffusion_rad2_s: float = 0.001
    ellipse_semi_major_axis_um: float = 2.0
    ellipse_semi_minor_axis_um: float = 0.35
    observation_lower_bounds_um: tuple[float, float] = (0.0, 0.0)
    observation_upper_bounds_um: tuple[float, float] = (10.0, 10.0)
    initial_centers_um: tuple[tuple[float, float], ...] | None = None
    equilibrium_centers_um: tuple[tuple[float, float], ...] | None = None
    initial_orientations_rad: tuple[float, ...] | None = None
    minimum_initial_center_separation_um: float = 1.5
    max_initialization_attempts: int = 10_000

    model_id: ClassVar[str] = "independent_dynamic_chromosomes_2d"
    local_dimension: ClassVar[int] = 2
    lab_dimension: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if (
            not isinstance(self.instance_count, int)
            or isinstance(self.instance_count, bool)
            or self.instance_count <= 0
        ):
            raise ValueError("instance_count must be a positive integer")
        rates_and_geometry = np.array(
            [
                self.chromosome_diffusion_coefficient_um2_s,
                self.chromosome_relaxation_rate_s,
                self.rotational_diffusion_rad2_s,
                self.ellipse_semi_major_axis_um,
                self.ellipse_semi_minor_axis_um,
                self.minimum_initial_center_separation_um,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(rates_and_geometry).all():
            raise ValueError("multi-chromosome parameters must be finite")
        if np.any(rates_and_geometry[:3] < 0.0):
            raise ValueError("chromosome rates must be nonnegative")
        if np.any(rates_and_geometry[3:5] <= 0.0):
            raise ValueError("ellipse semi-axes must be positive")
        if self.minimum_initial_center_separation_um < 0.0:
            raise ValueError("minimum center separation must be nonnegative")
        if (
            not isinstance(self.max_initialization_attempts, int)
            or isinstance(self.max_initialization_attempts, bool)
            or self.max_initialization_attempts <= 0
        ):
            raise ValueError("max_initialization_attempts must be positive")

        lower = np.asarray(self.observation_lower_bounds_um, dtype=np.float64)
        upper = np.asarray(self.observation_upper_bounds_um, dtype=np.float64)
        if lower.shape != (2,) or upper.shape != (2,):
            raise ValueError("observation bounds must each contain x and y")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("observation bounds must be finite")
        if np.any(upper <= lower):
            raise ValueError("observation upper bounds must exceed lower bounds")

        self._validate_centers(self.initial_centers_um, "initial_centers_um")
        self._validate_centers(
            self.equilibrium_centers_um, "equilibrium_centers_um"
        )
        if self.initial_orientations_rad is not None:
            orientations = np.asarray(
                self.initial_orientations_rad, dtype=np.float64
            )
            if orientations.shape != (self.instance_count,):
                raise ValueError(
                    "initial_orientations_rad must match instance_count"
                )
            if not np.isfinite(orientations).all():
                raise ValueError("initial orientations must be finite")

    def _validate_centers(
        self,
        centers_um: tuple[tuple[float, float], ...] | None,
        name: str,
    ) -> None:
        if centers_um is None:
            return
        centers = np.asarray(centers_um, dtype=np.float64)
        if centers.shape != (self.instance_count, 2):
            raise ValueError(f"{name} must have shape (instance_count, 2)")
        if not np.isfinite(centers).all():
            raise ValueError(f"{name} must be finite")

    def initialize_state(
        self, rng: np.random.Generator
    ) -> MultiChromosomeState2D:
        if self.initial_orientations_rad is None:
            orientations = rng.uniform(-np.pi, np.pi, self.instance_count)
        else:
            orientations = np.asarray(
                self.initial_orientations_rad, dtype=np.float64
            )
        if self.initial_centers_um is None:
            centers = self._sample_initial_centers(orientations, rng)
        else:
            centers = np.asarray(self.initial_centers_um, dtype=np.float64)
        if self.equilibrium_centers_um is None:
            equilibrium_centers = centers.copy()
        else:
            equilibrium_centers = np.asarray(
                self.equilibrium_centers_um, dtype=np.float64
            )
        poses = tuple(
            RigidPose2D(
                center_um=(float(center[0]), float(center[1])),
                orientation_rad=float(orientation),
            )
            for center, orientation in zip(centers, orientations, strict=True)
        )
        equilibrium = tuple(
            (float(center[0]), float(center[1]))
            for center in equilibrium_centers
        )
        return MultiChromosomeState2D(poses, equilibrium)

    def _sample_initial_centers(
        self,
        orientations_rad: NDArray[np.float64],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        lower = np.asarray(self.observation_lower_bounds_um, dtype=np.float64)
        upper = np.asarray(self.observation_upper_bounds_um, dtype=np.float64)
        semi_major = self.ellipse_semi_major_axis_um
        semi_minor = self.ellipse_semi_minor_axis_um
        centers: list[NDArray[np.float64]] = []
        for orientation in orientations_rad:
            cosine = np.cos(orientation)
            sine = np.sin(orientation)
            extents = np.array(
                [
                    np.sqrt((semi_major * cosine) ** 2 + (semi_minor * sine) ** 2),
                    np.sqrt((semi_major * sine) ** 2 + (semi_minor * cosine) ** 2),
                ],
                dtype=np.float64,
            )
            center_lower = lower + extents
            center_upper = upper - extents
            if np.any(center_upper < center_lower):
                raise ValueError(
                    "an oriented chromosome ellipse does not fit inside the "
                    "observation bounds"
                )
            for _ in range(self.max_initialization_attempts):
                candidate = rng.uniform(center_lower, center_upper)
                if all(
                    np.linalg.norm(candidate - existing)
                    >= self.minimum_initial_center_separation_um
                    for existing in centers
                ):
                    centers.append(candidate)
                    break
            else:
                raise RuntimeError(
                    "could not place all chromosome centers; reduce the count, "
                    "ellipse axes, or minimum separation"
                )
        return np.asarray(centers, dtype=np.float64)

    def step_state(
        self,
        state: MultiChromosomeState2D,
        frame_interval_s: float,
        rngs: tuple[np.random.Generator, ...],
    ) -> MultiChromosomeState2D:
        if frame_interval_s <= 0.0:
            raise ValueError("frame_interval_s must be positive")
        if len(state.poses) != self.instance_count:
            raise ValueError("state pose count does not match instance_count")
        if len(rngs) != self.instance_count:
            raise ValueError("one pose RNG is required per chromosome")

        next_poses: list[RigidPose2D] = []
        diffusion = self.chromosome_diffusion_coefficient_um2_s
        relaxation = self.chromosome_relaxation_rate_s
        rotational_diffusion = self.rotational_diffusion_rad2_s
        for pose, equilibrium_center, rng in zip(
            state.poses,
            state.equilibrium_centers_um,
            rngs,
            strict=True,
        ):
            center_innovation = rng.normal(0.0, 1.0, size=2)
            orientation_innovation = float(rng.normal(0.0, 1.0))
            center = np.asarray(pose.center_um, dtype=np.float64)
            equilibrium = np.asarray(equilibrium_center, dtype=np.float64)
            if diffusion == 0.0:
                next_center = center.copy()
            elif relaxation == 0.0:
                next_center = center + (
                    np.sqrt(2.0 * diffusion * frame_interval_s)
                    * center_innovation
                )
            else:
                decay = np.exp(-relaxation * frame_interval_s)
                noise_std = np.sqrt(
                    diffusion
                    / relaxation
                    * -np.expm1(-2.0 * relaxation * frame_interval_s)
                )
                next_center = (
                    equilibrium
                    + decay * (center - equilibrium)
                    + noise_std * center_innovation
                )
            if rotational_diffusion == 0.0:
                next_orientation = pose.orientation_rad
            else:
                next_orientation = pose.orientation_rad + (
                    np.sqrt(2.0 * rotational_diffusion * frame_interval_s)
                    * orientation_innovation
                )
            next_poses.append(
                RigidPose2D(
                    center_um=(float(next_center[0]), float(next_center[1])),
                    orientation_rad=float(next_orientation),
                )
            )
        return MultiChromosomeState2D(
            tuple(next_poses), state.equilibrium_centers_um
        )

    def transform(
        self,
        local_positions_um: NDArray[np.float64],
        instance_ids: NDArray[np.int64],
        state: MultiChromosomeState2D,
    ) -> NDArray[np.float64]:
        local = np.asarray(local_positions_um, dtype=np.float64)
        assignments = np.asarray(instance_ids)
        if local.ndim != 2 or local.shape[1] != 2:
            raise ValueError("local_positions_um must have shape (particles, 2)")
        if assignments.shape != (local.shape[0],):
            raise ValueError("instance_ids must have shape (particles,)")
        if not np.issubdtype(assignments.dtype, np.integer):
            raise ValueError("instance_ids must be integers")
        if assignments.size and (
            assignments.min() < 0 or assignments.max() >= self.instance_count
        ):
            raise ValueError("instance_ids contains an out-of-range chromosome")
        if len(state.poses) != self.instance_count:
            raise ValueError("state pose count does not match instance_count")

        transformed = np.empty_like(local)
        for chromosome_id, pose in enumerate(state.poses):
            selected = assignments == chromosome_id
            if not np.any(selected):
                continue
            cosine = np.cos(pose.orientation_rad)
            sine = np.sin(pose.orientation_rad)
            rotation = np.array(
                [[cosine, -sine], [sine, cosine]], dtype=np.float64
            )
            center = np.asarray(pose.center_um, dtype=np.float64)
            transformed[selected] = local[selected] @ rotation.T + center
        return transformed

    def parameters(self) -> Mapping[str, Any]:
        return {
            "instance_count": self.instance_count,
            "chromosome_diffusion_coefficient_um2_s": (
                self.chromosome_diffusion_coefficient_um2_s
            ),
            "chromosome_relaxation_rate_s": self.chromosome_relaxation_rate_s,
            "rotational_diffusion_rad2_s": self.rotational_diffusion_rad2_s,
            "ellipse_semi_major_axis_um": self.ellipse_semi_major_axis_um,
            "ellipse_semi_minor_axis_um": self.ellipse_semi_minor_axis_um,
            "observation_lower_bounds_um": list(
                self.observation_lower_bounds_um
            ),
            "observation_upper_bounds_um": list(
                self.observation_upper_bounds_um
            ),
            "initial_centers_um": (
                None
                if self.initial_centers_um is None
                else [list(center) for center in self.initial_centers_um]
            ),
            "equilibrium_centers_um": (
                None
                if self.equilibrium_centers_um is None
                else [list(center) for center in self.equilibrium_centers_um]
            ),
            "initial_orientations_rad": (
                None
                if self.initial_orientations_rad is None
                else list(self.initial_orientations_rad)
            ),
            "minimum_initial_center_separation_um": (
                self.minimum_initial_center_separation_um
            ),
            "max_initialization_attempts": self.max_initialization_attempts,
            "initial_pose_sampling": "seeded_random_inside_observation_bounds",
        }
