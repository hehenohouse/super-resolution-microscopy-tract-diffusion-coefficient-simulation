from __future__ import annotations

from srm_sim.models import (
    BrownianMotion2D,
    IdealOnStateObservation,
    ReflectingSquareBoundary,
    ThreeStateMarkovBlinking,
)
from srm_sim.scenario import SimulationScenario


BASELINE_SCENARIO_ID = "baseline_ideal_2d_brownian_markov"
BASELINE_SCENARIO_NAME = (
    "Baseline A - Ideal 2D Reflecting Brownian Motion with "
    "Three-State Markov Photophysics"
)


def create_baseline_scenario(
    *,
    diffusion_coefficient_um2_s: float = 0.5,
    field_size_um: float = 10.0,
    initial_on_fraction: float = 0.2,
    k_on_s: float = 1.0,
    k_off_s: float = 2.0,
    k_bleach_s: float = 0.08,
) -> SimulationScenario:
    return SimulationScenario(
        scenario_id=BASELINE_SCENARIO_ID,
        name=BASELINE_SCENARIO_NAME,
        motion=BrownianMotion2D(
            diffusion_coefficient_um2_s=diffusion_coefficient_um2_s
        ),
        boundary=ReflectingSquareBoundary(field_size_um=field_size_um),
        photophysics=ThreeStateMarkovBlinking(
            initial_on_fraction=initial_on_fraction,
            k_on_s=k_on_s,
            k_off_s=k_off_s,
            k_bleach_s=k_bleach_s,
        ),
        observation=IdealOnStateObservation(),
    )
