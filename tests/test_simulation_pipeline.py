from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from srm_sim import (
    SimulationConfig,
    SimulationScenario,
    create_free_diffusion_benchmark,
    create_projected_3d_free_diffusion_benchmark,
    create_reflecting_baseline,
    load_observations,
    save_animation,
    save_output_bundle,
    simulate,
)
from srm_sim.models import (
    BrownianMotion3D,
    IdealAxialSlabProjectionObservation,
    IdealOnStateObservation,
    ThreeStateMarkovBlinking,
    UnboundedSpace3D,
)


class ThreeDimensionalModelTests(unittest.TestCase):
    def test_brownian_variance_and_zero_diffusion(self) -> None:
        positions = np.zeros((100_000, 3), dtype=np.float64)
        dt = 0.05
        moved = BrownianMotion3D(0.5).step(
            positions, dt, np.random.default_rng(1)
        )
        np.testing.assert_allclose(moved.var(axis=0), 2.0 * 0.5 * dt, rtol=0.03)
        stationary = BrownianMotion3D(0.0).step(
            positions, dt, np.random.default_rng(2)
        )
        np.testing.assert_array_equal(stationary, positions)

    def test_unbounded_space_initialization_and_identity(self) -> None:
        boundary = UnboundedSpace3D((30.0, 30.0, 22.0))
        positions = boundary.sample_initial_positions(1_000, np.random.default_rng(3))
        self.assertEqual(positions.shape, (1_000, 3))
        self.assertTrue(np.all(positions >= boundary.bounds_um[:, 0]))
        self.assertTrue(np.all(positions <= boundary.bounds_um[:, 1]))
        outside = np.array([[-2.0, 31.0, 40.0]], dtype=np.float64)
        np.testing.assert_array_equal(boundary.apply(outside), outside)

    def test_axial_slab_visibility_and_projection(self) -> None:
        observation = IdealAxialSlabProjectionObservation(
            lower_bounds_um=(10.0, 10.0, 10.0),
            upper_bounds_um=(20.0, 20.0, 12.0),
        )
        positions = np.array(
            [
                [10.0, 10.0, 10.0],
                [20.0, 20.0, 12.0],
                [15.0, 15.0, 12.1],
                [9.9, 15.0, 11.0],
                [16.0, 17.0, 11.0],
            ],
            dtype=np.float64,
        )
        emitting = np.array([True, True, True, True, False])
        projected, visible = observation.observe(
            positions, emitting, np.random.default_rng(4)
        )
        np.testing.assert_array_equal(
            visible, np.array([True, True, False, False, False])
        )
        np.testing.assert_array_equal(projected[:2], positions[:2, :2])
        self.assertTrue(np.isnan(projected[2:]).all())
        self.assertEqual(projected.shape, (5, 2))
        self.assertEqual(observation.bounds_um.shape, (2, 2))

    def test_scenario_rejects_observation_input_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            SimulationScenario(
                scenario_id="invalid",
                name="invalid",
                motion=BrownianMotion3D(),
                boundary=UnboundedSpace3D(),
                photophysics=ThreeStateMarkovBlinking(),
                observation=IdealOnStateObservation(),
            )


class PipelineTests(unittest.TestCase):
    def test_three_dimensional_pipeline_and_export_boundary(self) -> None:
        scenario = create_projected_3d_free_diffusion_benchmark(
            observation_depth_um=10.0
        )
        config = SimulationConfig(
            n_particles=1_000,
            n_frames=5,
            frame_interval_s=0.05,
            random_seed=7,
        )
        result = simulate(config, scenario)
        repeated = simulate(config, scenario)

        self.assertEqual(result.positions_um.shape, (5, 1_000, 3))
        self.assertEqual(result.observed_positions_um.shape, (5, 1_000, 2))
        self.assertEqual(result.bounds_um.shape, (2, 2))
        np.testing.assert_array_equal(result.positions_um, repeated.positions_um)
        np.testing.assert_array_equal(result.states, repeated.states)
        np.testing.assert_array_equal(
            result.observed_positions_um, repeated.observed_positions_um
        )
        np.testing.assert_array_equal(
            result.observed_positions_um[result.visible],
            result.positions_um[result.visible, :2],
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = save_output_bundle(result, directory)
            observations = load_observations(paths.observations)
            self.assertEqual(observations.positions_um.shape[1], 2)
            with np.load(paths.observations, allow_pickle=False) as public:
                self.assertNotIn("particle_ids", public.files)
                self.assertNotIn("states", public.files)
                self.assertNotIn("detection_particle_ids", public.files)
                self.assertEqual(public["bounds_um"].shape, (2, 2))
            with np.load(paths.ground_truth, allow_pickle=False) as private:
                self.assertEqual(private["positions_um"].shape, (5, 1_000, 3))
                self.assertEqual(private["detection_true_positions_um"].shape[1], 3)
                self.assertEqual(int(private["physical_dimension"]), 3)
                self.assertEqual(int(private["observation_dimension"]), 2)
                np.testing.assert_array_equal(
                    observations.positions_um,
                    private["detection_true_positions_um"][:, :2],
                )

    def test_zero_detection_export_dimensions(self) -> None:
        scenario = create_projected_3d_free_diffusion_benchmark(
            initial_on_fraction=0.0,
            k_on_s=0.0,
            k_off_s=0.0,
            k_bleach_s=0.0,
        )
        result = simulate(
            SimulationConfig(n_particles=20, n_frames=2, random_seed=9), scenario
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = save_output_bundle(result, directory)
            observations = load_observations(paths.observations)
            self.assertEqual(observations.positions_um.shape, (0, 2))
            with np.load(paths.ground_truth, allow_pickle=False) as private:
                self.assertEqual(
                    private["detection_true_positions_um"].shape, (0, 3)
                )

    def test_existing_two_dimensional_scenarios_are_preserved(self) -> None:
        config = SimulationConfig(n_particles=30, n_frames=3, random_seed=11)
        for scenario in (
            create_reflecting_baseline(),
            create_free_diffusion_benchmark(),
        ):
            with self.subTest(scenario=scenario.scenario_id):
                result = simulate(config, scenario)
                self.assertEqual(result.positions_um.shape, (3, 30, 2))
                self.assertEqual(result.observed_positions_um.shape, (3, 30, 2))
                self.assertEqual(result.bounds_um.shape, (2, 2))

    def test_animation_accepts_2d_and_3d_truth(self) -> None:
        config = SimulationConfig(n_particles=30, n_frames=2, random_seed=13)
        scenarios = (
            create_reflecting_baseline(),
            create_projected_3d_free_diffusion_benchmark(
                observation_depth_um=10.0
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, scenario in enumerate(scenarios):
                output = Path(directory) / f"preview-{index}.gif"
                save_animation(simulate(config, scenario), output, dpi=30)
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
