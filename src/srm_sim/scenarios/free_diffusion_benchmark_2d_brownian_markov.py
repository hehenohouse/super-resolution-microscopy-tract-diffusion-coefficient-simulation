from __future__ import annotations

from srm_sim.models import (
    BrownianMotion2D,
    IdealOnStateObservation,
    ThreeStateMarkovBlinking,
    UnboundedPlane2D,
)
from srm_sim.scenario import SimulationScenario


FREE_DIFFUSION_BENCHMARK_ID = "free_diffusion_benchmark_2d_brownian_markov"
FREE_DIFFUSION_BENCHMARK_NAME = (
    "Free-Diffusion Benchmark - Unbounded 2D Brownian Motion with a Finite "
    "Observation Window and Three-State Markov Photophysics"
)


def create_free_diffusion_benchmark(
    *,
    diffusion_coefficient_um2_s: float = 0.5,
    initialization_size_um: float = 30.0,
    observation_window_size_um: float = 10.0,
    initial_on_fraction: float = 0.2,
    k_on_s: float = 1.0,
    k_off_s: float = 2.0,
    k_bleach_s: float = 0.08,
) -> SimulationScenario:
    if observation_window_size_um <= 0:
        raise ValueError("observation_window_size_um must be positive")
    if observation_window_size_um >= initialization_size_um:
        raise ValueError(
            "observation_window_size_um must be smaller than "
            "initialization_size_um"
        )

    window_lower_um = (initialization_size_um - observation_window_size_um) / 2.0
    window_upper_um = window_lower_um + observation_window_size_um

    return SimulationScenario(
        scenario_id=FREE_DIFFUSION_BENCHMARK_ID,
        name=FREE_DIFFUSION_BENCHMARK_NAME,
        motion=BrownianMotion2D(
            diffusion_coefficient_um2_s=diffusion_coefficient_um2_s
        ),
        boundary=UnboundedPlane2D(
            initialization_size_um=initialization_size_um
        ),
        photophysics=ThreeStateMarkovBlinking(
            initial_on_fraction=initial_on_fraction,
            k_on_s=k_on_s,
            k_off_s=k_off_s,
            k_bleach_s=k_bleach_s,
        ),
        observation=IdealOnStateObservation(
            lower_bounds_um=(window_lower_um, window_lower_um),
            upper_bounds_um=(window_upper_um, window_upper_um),
        ),
    )
