from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class IdealOnStateObservation:
    model_id: ClassVar[str] = "ideal_emitter_localization"

    def observe(
        self,
        positions_um: NDArray[np.float64],
        emitting: NDArray[np.bool_],
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
        del rng
        observed_positions_um = np.full_like(positions_um, np.nan)
        observed_positions_um[emitting] = positions_um[emitting]
        return observed_positions_um, emitting.copy()

    def parameters(self) -> Mapping[str, Any]:
        return {}
