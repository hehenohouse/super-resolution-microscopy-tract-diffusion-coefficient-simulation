from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class BrownianMotion2D:
    diffusion_coefficient_um2_s: float = 0.5

    model_id: ClassVar[str] = "homogeneous_brownian_motion_2d"
    dimension: ClassVar[int] = 2

    def __post_init__(self) -> None:
        if self.diffusion_coefficient_um2_s < 0:
            raise ValueError("diffusion_coefficient_um2_s cannot be negative")

    def step(
        self,
        positions_um: NDArray[np.float64],
        frame_interval_s: float,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        step_std_um = np.sqrt(
            2.0 * self.diffusion_coefficient_um2_s * frame_interval_s
        )
        displacement = rng.normal(
            loc=0.0,
            scale=step_std_um,
            size=positions_um.shape,
        )
        return positions_um + displacement

    def parameters(self) -> Mapping[str, Any]:
        return {
            "diffusion_coefficient_um2_s": self.diffusion_coefficient_um2_s
        }


@dataclass(frozen=True)
class BrownianMotion3D:
    diffusion_coefficient_um2_s: float = 0.5

    model_id: ClassVar[str] = "homogeneous_brownian_motion_3d"
    dimension: ClassVar[int] = 3

    def __post_init__(self) -> None:
        if self.diffusion_coefficient_um2_s < 0:
            raise ValueError("diffusion_coefficient_um2_s cannot be negative")

    def step(
        self,
        positions_um: NDArray[np.float64],
        frame_interval_s: float,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        step_std_um = np.sqrt(
            2.0 * self.diffusion_coefficient_um2_s * frame_interval_s
        )
        displacement = rng.normal(
            loc=0.0,
            scale=step_std_um,
            size=positions_um.shape,
        )
        return positions_um + displacement

    def parameters(self) -> Mapping[str, Any]:
        return {
            "diffusion_coefficient_um2_s": self.diffusion_coefficient_um2_s
        }
