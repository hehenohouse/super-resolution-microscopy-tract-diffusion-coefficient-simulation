from pathlib import Path

from srm_sim import (
    SimulationConfig,
    create_baseline_scenario,
    save_animation,
    save_output_bundle,
    simulate,
)


def main() -> None:
    run_config = SimulationConfig(
        n_particles=100,
        n_frames=200,
        frame_interval_s=0.05,
        random_seed=7,
    )
    scenario = create_baseline_scenario(
        diffusion_coefficient_um2_s=0.5,
        field_size_um=10.0,
        initial_on_fraction=0.2,
        k_on_s=1.0,
        k_off_s=2.0,
        k_bleach_s=0.08,
    )
    result = simulate(run_config, scenario)

    output_directory = Path("outputs")
    output_paths = save_output_bundle(
        result,
        output_directory,
        detection_order_seed=17,
    )
    animation_path = save_animation(
        result, output_directory / "simulation_preview.gif"
    )

    metadata = scenario.metadata()
    print("SRM simulation complete")
    print(f"  scenario: {scenario.scenario_id}")
    print(f"  name: {scenario.name}")
    for role, component in metadata["components"].items():
        print(f"  {role}: {component['model_id']}")
    print(f"  particles: {run_config.n_particles}")
    print(f"  frames: {run_config.n_frames}")
    print(f"  final emitting: {int(result.emitting[-1].sum())}")
    print(f"  final active: {int(result.active[-1].sum())}")
    print(f"  final visible: {int(result.visible[-1].sum())}")
    print(f"  tracking input: {output_paths.observations}")
    print(f"  private ground truth: {output_paths.ground_truth}")
    print(f"  animation: {animation_path}")


if __name__ == "__main__":
    main()
