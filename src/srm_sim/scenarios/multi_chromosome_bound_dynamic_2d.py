from __future__ import annotations

from srm_sim.models import (
    BrownianMotion2D,
    GaussianLocalizationObservation,
    IndependentChromosomesScene2D,
    ReflectingEllipseBoundary,
    ThreeStateMarkovBlinking,
)
from srm_sim.scenario import SimulationScenario


MULTI_CHROMOSOME_BOUND_DYNAMIC_2D_ID = (
    "multi_chromosome_bound_dynamic_2d"
)
MULTI_CHROMOSOME_BOUND_DYNAMIC_2D_NAME = (
    "Multi-Chromosome-Bound Dynamic 2D - Brownian Proteins in "
    "Independent Moving Effective Elliptical Masks"
)


def create_multi_chromosome_bound_dynamic_2d(
    *,
    n_chromosomes: int = 6,
    protein_diffusion_coefficient_um2_s: float = 0.1,
    chromosome_diffusion_coefficient_um2_s: float = 0.001,
    chromosome_relaxation_rate_s: float = 1.0 / 30.0,
    rotational_diffusion_rad2_s: float = 0.001,
    ellipse_semi_major_axis_um: float = 2.0,
    ellipse_semi_minor_axis_um: float = 0.35,
    initial_centers_um: tuple[tuple[float, float], ...] | None = None,
    equilibrium_centers_um: tuple[tuple[float, float], ...] | None = None,
    initial_orientations_rad: tuple[float, ...] | None = None,
    minimum_initial_center_separation_um: float = 1.5,
    max_initialization_attempts: int = 10_000,
    observation_lower_bounds_um: tuple[float, float] = (0.0, 0.0),
    observation_upper_bounds_um: tuple[float, float] = (10.0, 10.0),
    localization_sigma_um: float = 0.03,
    initial_on_fraction: float = 0.2,
    k_on_s: float = 1.0,
    k_off_s: float = 2.0,
    k_bleach_s: float = 0.08,
) -> SimulationScenario:
    """Compose independently moving chromosomes with permanent proteins."""
    return SimulationScenario(
        scenario_id=MULTI_CHROMOSOME_BOUND_DYNAMIC_2D_ID,
        name=MULTI_CHROMOSOME_BOUND_DYNAMIC_2D_NAME,
        motion=BrownianMotion2D(
            diffusion_coefficient_um2_s=protein_diffusion_coefficient_um2_s
        ),
        boundary=ReflectingEllipseBoundary(
            semi_major_axis_um=ellipse_semi_major_axis_um,
            semi_minor_axis_um=ellipse_semi_minor_axis_um,
        ),
        photophysics=ThreeStateMarkovBlinking(
            initial_on_fraction=initial_on_fraction,
            k_on_s=k_on_s,
            k_off_s=k_off_s,
            k_bleach_s=k_bleach_s,
        ),
        observation=GaussianLocalizationObservation(
            lower_bounds_um=observation_lower_bounds_um,
            upper_bounds_um=observation_upper_bounds_um,
            localization_sigma_um=localization_sigma_um,
        ),
        multi_scene=IndependentChromosomesScene2D(
            instance_count=n_chromosomes,
            chromosome_diffusion_coefficient_um2_s=(
                chromosome_diffusion_coefficient_um2_s
            ),
            chromosome_relaxation_rate_s=chromosome_relaxation_rate_s,
            rotational_diffusion_rad2_s=rotational_diffusion_rad2_s,
            ellipse_semi_major_axis_um=ellipse_semi_major_axis_um,
            ellipse_semi_minor_axis_um=ellipse_semi_minor_axis_um,
            observation_lower_bounds_um=observation_lower_bounds_um,
            observation_upper_bounds_um=observation_upper_bounds_um,
            initial_centers_um=initial_centers_um,
            equilibrium_centers_um=equilibrium_centers_um,
            initial_orientations_rad=initial_orientations_rad,
            minimum_initial_center_separation_um=(
                minimum_initial_center_separation_um
            ),
            max_initialization_attempts=max_initialization_attempts,
        ),
    )
