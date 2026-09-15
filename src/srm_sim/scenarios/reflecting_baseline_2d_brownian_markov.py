from __future__ import annotations

from srm_sim.models import (
    BrownianMotion2D,
    IdealOnStateObservation,
    ReflectingSquareBoundary,
    ThreeStateMarkovBlinking,
)
from srm_sim.scenario import SimulationScenario


REFLECTING_BASELINE_ID = "reflecting_baseline_2d_brownian_markov"
REFLECTING_BASELINE_NAME = (
    "Reflecting Baseline - Ideal 2D Brownian Motion in a Reflecting Square "
    "with Three-State Markov Photophysics"
)


def create_reflecting_baseline(
    *,
    diffusion_coefficient_um2_s: float = 0.5,
    field_size_um: float = 10.0,
    initial_on_fraction: float = 0.2,
    k_on_s: float = 1.0,
    k_off_s: float = 2.0,
    k_bleach_s: float = 0.08,
) -> SimulationScenario:
    boundary = ReflectingSquareBoundary(field_size_um=field_size_um)
    return SimulationScenario(
        scenario_id=REFLECTING_BASELINE_ID,
        name=REFLECTING_BASELINE_NAME,
        motion=BrownianMotion2D(
            diffusion_coefficient_um2_s=diffusion_coefficient_um2_s
        ),
        boundary=boundary,
        photophysics=ThreeStateMarkovBlinking(
            initial_on_fraction=initial_on_fraction,
            k_on_s=k_on_s,
            k_off_s=k_off_s,
            k_bleach_s=k_bleach_s,
        ),
        observation=IdealOnStateObservation(
            lower_bounds_um=(0.0, 0.0),
            upper_bounds_um=(field_size_um, field_size_um),
        ),
    )
