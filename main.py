from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from srm_sim import (
    SimulationConfig,
    SimulationScenario,
    create_chromosome_bound_dynamic_2d,
    create_free_diffusion_benchmark,
    create_multi_chromosome_bound_dynamic_2d,
    create_projected_3d_free_diffusion_benchmark,
    create_reflecting_baseline,
    save_animation,
    save_output_bundle,
    simulate,
)


ScenarioBuilder = Callable[[], tuple[SimulationConfig, SimulationScenario]]


def build_chromosome_bound_dynamic_2d(
) -> tuple[SimulationConfig, SimulationScenario]:
    run_config = SimulationConfig(
        n_particles=100,
        n_frames=200,
        frame_interval_s=0.05,
        random_seed=7,
    )
    scenario = create_chromosome_bound_dynamic_2d(
        protein_diffusion_coefficient_um2_s=0.005,
        chromosome_diffusion_coefficient_um2_s=0.001,
        chromosome_relaxation_rate_s=1.0 / 30.0,
        rotational_diffusion_rad2_s=0.001,
        ellipse_semi_major_axis_um=2.0,
        ellipse_semi_minor_axis_um=0.35,
        localization_sigma_um=0.03,
    )
    return run_config, scenario


def build_multi_chromosome_bound_dynamic_2d(
) -> tuple[SimulationConfig, SimulationScenario]:
    run_config = SimulationConfig(
        n_particles=120,
        n_frames=200,
        frame_interval_s=0.05,
        random_seed=7,
    )
    scenario = create_multi_chromosome_bound_dynamic_2d(
        n_chromosomes=6,
        protein_diffusion_coefficient_um2_s=0.1,
        chromosome_diffusion_coefficient_um2_s=0.001,
        chromosome_relaxation_rate_s=1.0 / 30.0,
        rotational_diffusion_rad2_s=0.001,
        ellipse_semi_major_axis_um=2.0,
        ellipse_semi_minor_axis_um=0.35,
        localization_sigma_um=0.03,
    )
    return run_config, scenario


def build_reflecting_baseline() -> tuple[SimulationConfig, SimulationScenario]:
    run_config = SimulationConfig(
        n_particles=100,
        n_frames=200,
        frame_interval_s=0.05,
        random_seed=7,
    )
    scenario = create_reflecting_baseline(
        diffusion_coefficient_um2_s=0.5,
        field_size_um=10.0,
        initial_on_fraction=0.2,
        k_on_s=1.0,
        k_off_s=2.0,
        k_bleach_s=0.08,
    )
    return run_config, scenario


def build_free_diffusion_benchmark(
) -> tuple[SimulationConfig, SimulationScenario]:
    run_config = SimulationConfig(
        n_particles=900,
        n_frames=200,
        frame_interval_s=0.05,
        random_seed=7,
    )
    scenario = create_free_diffusion_benchmark(
        diffusion_coefficient_um2_s=0.5,
        initialization_size_um=30.0,
        observation_window_size_um=10.0,
        initial_on_fraction=0.2,
        k_on_s=1.0,
        k_off_s=2.0,
        k_bleach_s=0.08,
    )
    return run_config, scenario


def build_projected_3d_free_diffusion_benchmark(
) -> tuple[SimulationConfig, SimulationScenario]:
    run_config = SimulationConfig(
        n_particles=5_000,
        n_frames=200,
        frame_interval_s=0.05,
        random_seed=7,
    )
    scenario = create_projected_3d_free_diffusion_benchmark(
        diffusion_coefficient_um2_s=0.5,
        initialization_lateral_size_um=30.0,
        initialization_axial_size_um=22.0,
        observation_window_size_um=10.0,
        observation_depth_um=2.0,
        initial_on_fraction=0.2,
        k_on_s=1.0,
        k_off_s=2.0,
        k_bleach_s=0.08,
    )
    return run_config, scenario


SCENARIO_BUILDERS: dict[str, ScenarioBuilder] = {
    "3d": build_projected_3d_free_diffusion_benchmark,
    "chromosome": build_chromosome_bound_dynamic_2d,
    "chromosomes": build_multi_chromosome_bound_dynamic_2d,
    "reflecting": build_reflecting_baseline,
    "free": build_free_diffusion_benchmark,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an SRM tracking simulation scenario."
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIO_BUILDERS,
        default="3d",
        help="Scenario to run (default: 3d).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
        help="Root directory for generated scenario outputs.",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        help="Skip GIF generation for faster numerical runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config, scenario = SCENARIO_BUILDERS[args.scenario]()
    result = simulate(run_config, scenario)

    output_directory = args.output_root / scenario.scenario_id
    output_paths = save_output_bundle(
        result,
        output_directory,
        detection_order_seed=17,
    )
    animation_path = None
    if not args.no_animation:
        animation_path = save_animation(
            result, output_directory / "simulation_preview.gif"
        )

    metadata = scenario.metadata()
    final_in_region = result.in_observation_region[-1]
    print("SRM simulation complete")
    print(f"  scenario: {scenario.scenario_id}")
    print(f"  name: {scenario.name}")
    for role, component in metadata["components"].items():
        print(f"  {role}: {component['model_id']}")
    print(f"  particles: {run_config.n_particles}")
    print(f"  frames: {run_config.n_frames}")
    print(f"  final particles in FOV: {int(final_in_region.sum())}")
    print(
        "  final emitting in FOV: "
        f"{int((result.emitting[-1] & final_in_region).sum())}"
    )
    print(f"  final active: {int(result.active[-1].sum())}")
    print(f"  final visible: {int(result.visible[-1].sum())}")
    print(f"  tracking input: {output_paths.observations}")
    print(f"  private ground truth: {output_paths.ground_truth}")
    if animation_path is not None:
        print(f"  animation: {animation_path}")


if __name__ == "__main__":
    main()
