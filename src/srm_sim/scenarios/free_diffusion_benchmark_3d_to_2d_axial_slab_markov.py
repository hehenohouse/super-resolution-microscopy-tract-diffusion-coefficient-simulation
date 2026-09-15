from __future__ import annotations

from srm_sim.models import (
    BrownianMotion3D,
    IdealAxialSlabProjectionObservation,
    ThreeStateMarkovBlinking,
    UnboundedSpace3D,
)
from srm_sim.scenario import SimulationScenario


PROJECTED_3D_FREE_DIFFUSION_BENCHMARK_ID = (
    "free_diffusion_benchmark_3d_to_2d_axial_slab_markov"
)
PROJECTED_3D_FREE_DIFFUSION_BENCHMARK_NAME = (
    "Free-Diffusion Benchmark - Unbounded 3D Brownian Motion with an Ideal "
    "2D Axial-Slab Projection and Three-State Markov Photophysics"
)


def create_projected_3d_free_diffusion_benchmark(
    *,
    diffusion_coefficient_um2_s: float = 0.5,
    initialization_lateral_size_um: float = 30.0,
    initialization_axial_size_um: float = 22.0,
    observation_window_size_um: float = 10.0,
    observation_depth_um: float = 2.0,
    initial_on_fraction: float = 0.2,
    k_on_s: float = 1.0,
    k_off_s: float = 2.0,
    k_bleach_s: float = 0.08,
) -> SimulationScenario:
    if observation_window_size_um <= 0:
        raise ValueError("observation_window_size_um must be positive")
    if observation_depth_um <= 0:
        raise ValueError("observation_depth_um must be positive")
    if observation_window_size_um >= initialization_lateral_size_um:
        raise ValueError(
            "observation_window_size_um must be smaller than the lateral "
            "initialization size"
        )
    if observation_depth_um >= initialization_axial_size_um:
        raise ValueError(
            "observation_depth_um must be smaller than the axial "
            "initialization size"
        )

    lateral_lower_um = (
        initialization_lateral_size_um - observation_window_size_um
    ) / 2.0
    lateral_upper_um = lateral_lower_um + observation_window_size_um
    axial_lower_um = (
        initialization_axial_size_um - observation_depth_um
    ) / 2.0
    axial_upper_um = axial_lower_um + observation_depth_um

    return SimulationScenario(
        scenario_id=PROJECTED_3D_FREE_DIFFUSION_BENCHMARK_ID,
        name=PROJECTED_3D_FREE_DIFFUSION_BENCHMARK_NAME,
        motion=BrownianMotion3D(
            diffusion_coefficient_um2_s=diffusion_coefficient_um2_s
        ),
        boundary=UnboundedSpace3D(
            initialization_lengths_um=(
                initialization_lateral_size_um,
                initialization_lateral_size_um,
                initialization_axial_size_um,
            )
        ),
        photophysics=ThreeStateMarkovBlinking(
            initial_on_fraction=initial_on_fraction,
            k_on_s=k_on_s,
            k_off_s=k_off_s,
            k_bleach_s=k_bleach_s,
        ),
        observation=IdealAxialSlabProjectionObservation(
            lower_bounds_um=(
                lateral_lower_um,
                lateral_lower_um,
                axial_lower_um,
            ),
            upper_bounds_um=(
                lateral_upper_um,
                lateral_upper_um,
                axial_upper_um,
            ),
        ),
    )
