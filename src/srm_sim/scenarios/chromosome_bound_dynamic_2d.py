from __future__ import annotations

from srm_sim.models import (
    BrownianMotion2D,
    DynamicChromosomeScene2D,
    GaussianLocalizationObservation,
    ReflectingEllipseBoundary,
    ThreeStateMarkovBlinking,
)
from srm_sim.scenario import SimulationScenario


CHROMOSOME_BOUND_DYNAMIC_2D_ID = "chromosome_bound_dynamic_2d"
CHROMOSOME_BOUND_DYNAMIC_2D_NAME = (
    "Chromosome-Bound Dynamic 2D - Brownian Proteins in an Effective "
    "Elliptical Mask with Shared Chromosome Translation and Rotation"
)


def create_chromosome_bound_dynamic_2d(
    *,
    protein_diffusion_coefficient_um2_s: float = 0.005,
    chromosome_diffusion_coefficient_um2_s: float = 0.001,
    chromosome_relaxation_rate_s: float = 1.0 / 30.0,
    rotational_diffusion_rad2_s: float = 0.001,
    ellipse_semi_major_axis_um: float = 2.0,
    ellipse_semi_minor_axis_um: float = 0.35,
    equilibrium_center_um: tuple[float, float] = (5.0, 5.0),
    initial_center_um: tuple[float, float] = (5.0, 5.0),
    initial_orientation_rad: float = 0.0,
    observation_lower_bounds_um: tuple[float, float] = (0.0, 0.0),
    observation_upper_bounds_um: tuple[float, float] = (10.0, 10.0),
    localization_sigma_um: float = 0.03,
    initial_on_fraction: float = 0.2,
    k_on_s: float = 1.0,
    k_off_s: float = 2.0,
    k_bleach_s: float = 0.08,
) -> SimulationScenario:
    """Compose the chromosome-bound simulation from reusable components."""
    return SimulationScenario(
        scenario_id=CHROMOSOME_BOUND_DYNAMIC_2D_ID,
        name=CHROMOSOME_BOUND_DYNAMIC_2D_NAME,
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
        scene=DynamicChromosomeScene2D(
            chromosome_diffusion_coefficient_um2_s=(
                chromosome_diffusion_coefficient_um2_s
            ),
            chromosome_relaxation_rate_s=chromosome_relaxation_rate_s,
            rotational_diffusion_rad2_s=rotational_diffusion_rad2_s,
            equilibrium_center_um=equilibrium_center_um,
            initial_center_um=initial_center_um,
            initial_orientation_rad=initial_orientation_rad,
        ),
    )
