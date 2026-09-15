from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar, Mapping

import numpy as np
from numpy.typing import NDArray


class FluorophoreState(IntEnum):
    OFF = 0
    ON = 1
    BLEACHED = 2


@dataclass(frozen=True)
class ThreeStateMarkovBlinking:
    initial_on_fraction: float = 0.2
    k_on_s: float = 1.0
    k_off_s: float = 2.0
    k_bleach_s: float = 0.08

    model_id: ClassVar[str] = "three_state_markov_blinking_with_bleaching"

    def __post_init__(self) -> None:
        if not 0.0 <= self.initial_on_fraction <= 1.0:
            raise ValueError("initial_on_fraction must be between 0 and 1")
        if min(self.k_on_s, self.k_off_s, self.k_bleach_s) < 0:
            raise ValueError("state transition rates cannot be negative")

    def initialize_states(
        self, n_particles: int, rng: np.random.Generator
    ) -> NDArray[np.uint8]:
        return np.where(
            rng.random(n_particles) < self.initial_on_fraction,
            FluorophoreState.ON,
            FluorophoreState.OFF,
        ).astype(np.uint8)

    def step(
        self,
        states: NDArray[np.uint8],
        frame_interval_s: float,
        rng: np.random.Generator,
    ) -> NDArray[np.uint8]:
        current_states = states.copy()

        p_turn_on = 1.0 - np.exp(-self.k_on_s * frame_interval_s)
        off_particles = states == FluorophoreState.OFF
        turn_on = off_particles & (rng.random(states.size) < p_turn_on)
        current_states[turn_on] = FluorophoreState.ON

        on_exit_rate = self.k_off_s + self.k_bleach_s
        p_leave_on = 1.0 - np.exp(-on_exit_rate * frame_interval_s)
        bleach_given_exit = (
            self.k_bleach_s / on_exit_rate if on_exit_rate > 0.0 else 0.0
        )

        on_particles = states == FluorophoreState.ON
        leave_on = on_particles & (rng.random(states.size) < p_leave_on)
        bleach = leave_on & (rng.random(states.size) < bleach_given_exit)
        current_states[leave_on] = FluorophoreState.OFF
        current_states[bleach] = FluorophoreState.BLEACHED

        return current_states

    def emitting_mask(
        self, states: NDArray[np.uint8]
    ) -> NDArray[np.bool_]:
        return states == FluorophoreState.ON

    def active_mask(
        self, states: NDArray[np.uint8]
    ) -> NDArray[np.bool_]:
        return states != FluorophoreState.BLEACHED

    def parameters(self) -> Mapping[str, Any]:
        return {
            "initial_on_fraction": self.initial_on_fraction,
            "k_on_s": self.k_on_s,
            "k_off_s": self.k_off_s,
            "k_bleach_s": self.k_bleach_s,
        }
