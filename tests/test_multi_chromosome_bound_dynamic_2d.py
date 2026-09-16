from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from main import build_multi_chromosome_bound_dynamic_2d
from srm_sim import (
    MULTI_CHROMOSOME_BOUND_DYNAMIC_2D_ID,
    SimulationConfig,
    create_multi_chromosome_bound_dynamic_2d,
    save_animation,
    save_output_bundle,
    simulate,
)
from srm_sim.models import IndependentChromosomesScene2D
from srm_sim.visualization import _protein_trail_data


class MultiChromosomeScenarioTests(unittest.TestCase):
    def test_stronger_default_local_diffusion(self) -> None:
        scenario = create_multi_chromosome_bound_dynamic_2d()
        self.assertEqual(
            scenario.motion.parameters()["diffusion_coefficient_um2_s"],
            0.1,
        )
        run_config, cli_scenario = build_multi_chromosome_bound_dynamic_2d()
        self.assertEqual(run_config.frame_interval_s, 0.05)
        self.assertEqual(
            cli_scenario.motion.parameters()["diffusion_coefficient_um2_s"],
            0.1,
        )

        positions = np.zeros((200_000, 2), dtype=np.float64)
        moved = scenario.motion.step(
            positions, run_config.frame_interval_s, np.random.default_rng(20)
        )
        expected_std_um = np.sqrt(2.0 * 0.1 * 0.05)
        np.testing.assert_allclose(
            moved.std(axis=0), expected_std_um, rtol=0.01
        )

    def test_private_protein_trail_selection_is_bounded_and_deterministic(
        self,
    ) -> None:
        result = simulate(
            SimulationConfig(
                n_particles=24,
                n_frames=4,
                frame_interval_s=0.05,
                random_seed=20,
            ),
            create_multi_chromosome_bound_dynamic_2d(n_chromosomes=4),
        )
        trail_data = _protein_trail_data(result, duration_s=0.5)
        assert trail_data is not None
        particle_ids, chromosome_ids, trail_frame_count = trail_data
        np.testing.assert_array_equal(
            particle_ids, np.array([0, 1, 6, 7, 12, 13, 18, 19])
        )
        np.testing.assert_array_equal(
            chromosome_ids, np.repeat(np.arange(4), 2)
        )
        self.assertEqual(trail_frame_count, 10)

        repeated = _protein_trail_data(result, duration_s=0.5)
        assert repeated is not None
        np.testing.assert_array_equal(repeated[0], particle_ids)
        np.testing.assert_array_equal(repeated[1], chromosome_ids)
        self.assertEqual(repeated[2], trail_frame_count)

    def test_randomized_poses_assignments_confinement_and_reconstruction(self) -> None:
        scenario = create_multi_chromosome_bound_dynamic_2d()
        config = SimulationConfig(
            n_particles=62,
            n_frames=30,
            frame_interval_s=0.05,
            random_seed=21,
        )
        result = simulate(config, scenario)
        repeated = simulate(config, scenario)

        self.assertEqual(
            scenario.scenario_id, MULTI_CHROMOSOME_BOUND_DYNAMIC_2D_ID
        )
        self.assertEqual(result.particle_scene_ids.shape, (62,))
        self.assertEqual(result.scene_instance_centers_um.shape, (30, 6, 2))
        self.assertEqual(
            result.scene_instance_orientations_rad.shape, (30, 6)
        )
        counts = np.bincount(result.particle_scene_ids, minlength=6)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)
        self.assertGreater(
            np.unique(result.scene_instance_centers_um[0], axis=0).shape[0], 1
        )
        self.assertGreater(
            np.unique(result.scene_instance_orientations_rad[0]).size, 1
        )

        local = result.local_positions_um
        centers = result.scene_instance_centers_um
        orientations = result.scene_instance_orientations_rad
        assignments = result.particle_scene_ids
        assert local is not None and centers is not None
        assert orientations is not None and assignments is not None
        scaled_radius_squared = (
            np.square(local[..., 0] / 2.0)
            + np.square(local[..., 1] / 0.35)
        )
        self.assertLessEqual(float(scaled_radius_squared.max()), 1.0 + 1e-10)

        assigned_centers = centers[:, assignments]
        assigned_orientations = orientations[:, assignments]
        cosine = np.cos(assigned_orientations)
        sine = np.sin(assigned_orientations)
        reconstructed = np.empty_like(result.positions_um)
        reconstructed[..., 0] = (
            assigned_centers[..., 0]
            + cosine * local[..., 0]
            - sine * local[..., 1]
        )
        reconstructed[..., 1] = (
            assigned_centers[..., 1]
            + sine * local[..., 0]
            + cosine * local[..., 1]
        )
        np.testing.assert_allclose(reconstructed, result.positions_um, atol=1e-14)
        np.testing.assert_array_equal(result.positions_um, repeated.positions_um)
        np.testing.assert_array_equal(
            result.particle_scene_ids, repeated.particle_scene_ids
        )
        np.testing.assert_array_equal(
            centers, repeated.scene_instance_centers_um
        )
        np.testing.assert_array_equal(
            orientations, repeated.scene_instance_orientations_rad
        )

        for center, orientation in zip(
            centers[0], orientations[0], strict=True
        ):
            x_extent = np.sqrt(
                (2.0 * np.cos(orientation)) ** 2
                + (0.35 * np.sin(orientation)) ** 2
            )
            y_extent = np.sqrt(
                (2.0 * np.sin(orientation)) ** 2
                + (0.35 * np.cos(orientation)) ** 2
            )
            self.assertGreaterEqual(center[0] - x_extent, -1e-12)
            self.assertLessEqual(center[0] + x_extent, 10.0 + 1e-12)
            self.assertGreaterEqual(center[1] - y_extent, -1e-12)
            self.assertLessEqual(center[1] + y_extent, 10.0 + 1e-12)

    def test_multi_pose_controls_and_rng_isolation(self) -> None:
        config = SimulationConfig(
            n_particles=24,
            n_frames=15,
            frame_interval_s=0.05,
            random_seed=22,
        )
        translating = simulate(
            config,
            create_multi_chromosome_bound_dynamic_2d(
                n_chromosomes=4,
                protein_diffusion_coefficient_um2_s=0.0,
                chromosome_diffusion_coefficient_um2_s=0.001,
                rotational_diffusion_rad2_s=0.0,
            ),
        )
        translating_and_rotating = simulate(
            config,
            create_multi_chromosome_bound_dynamic_2d(
                n_chromosomes=4,
                protein_diffusion_coefficient_um2_s=0.0,
                chromosome_diffusion_coefficient_um2_s=0.001,
                rotational_diffusion_rad2_s=0.001,
            ),
        )
        rotating = simulate(
            config,
            create_multi_chromosome_bound_dynamic_2d(
                n_chromosomes=4,
                protein_diffusion_coefficient_um2_s=0.0,
                chromosome_diffusion_coefficient_um2_s=0.0,
                rotational_diffusion_rad2_s=0.001,
            ),
        )
        np.testing.assert_array_equal(
            translating.scene_instance_centers_um,
            translating_and_rotating.scene_instance_centers_um,
        )
        np.testing.assert_array_equal(
            rotating.scene_instance_orientations_rad,
            translating_and_rotating.scene_instance_orientations_rad,
        )

        static = simulate(
            config,
            create_multi_chromosome_bound_dynamic_2d(
                n_chromosomes=4,
                chromosome_diffusion_coefficient_um2_s=0.0,
                rotational_diffusion_rad2_s=0.0,
            ),
        )
        np.testing.assert_array_equal(
            static.scene_instance_centers_um,
            np.broadcast_to(
                static.scene_instance_centers_um[0],
                static.scene_instance_centers_um.shape,
            ),
        )
        np.testing.assert_array_equal(
            static.scene_instance_orientations_rad,
            np.broadcast_to(
                static.scene_instance_orientations_rad[0],
                static.scene_instance_orientations_rad.shape,
            ),
        )

    def test_random_placement_failure_is_clear(self) -> None:
        scene = IndependentChromosomesScene2D(
            instance_count=2,
            ellipse_semi_major_axis_um=2.0,
            ellipse_semi_minor_axis_um=0.35,
            observation_lower_bounds_um=(0.0, 0.0),
            observation_upper_bounds_um=(3.0, 3.0),
        )
        with self.assertRaisesRegex(ValueError, "does not fit"):
            scene.initialize_state(np.random.default_rng(23))

    def test_multi_export_is_private_and_versioned(self) -> None:
        result = simulate(
            SimulationConfig(n_particles=24, n_frames=4, random_seed=24),
            create_multi_chromosome_bound_dynamic_2d(
                n_chromosomes=4,
                initial_on_fraction=1.0,
                k_on_s=0.0,
                k_off_s=0.0,
                k_bleach_s=0.0,
            ),
        )
        expected_public_keys = {
            "format_version",
            "scenario_id",
            "observation_model",
            "position_unit",
            "time_unit",
            "n_frames",
            "frame_interval_s",
            "frame_times_s",
            "bounds_um",
            "detection_order_seed",
            "detection_ids",
            "frame_indices",
            "times_s",
            "positions_um",
        }
        required_private = {
            "particle_chromosome_ids",
            "chromosome_local_positions_um",
            "chromosome_centers_um",
            "chromosome_orientations_rad",
            "chromosome_ellipse_semi_axes_um",
            "chromosome_initialization_bounds_um",
            "observed_positions_um",
            "visible",
        }
        forbidden_public = required_private | {
            "particle_ids",
            "detection_particle_ids",
            "scene_model",
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = save_output_bundle(result, directory)
            with np.load(paths.observations, allow_pickle=False) as public:
                self.assertEqual(set(public.files), expected_public_keys)
                self.assertTrue(forbidden_public.isdisjoint(public.files))
                replay_rng = np.random.default_rng(
                    int(public["detection_order_seed"])
                )
                replayed_particle_ids = []
                for _ in range(result.config.n_frames):
                    frame_slots = np.arange(
                        result.config.n_particles, dtype=np.int64
                    )
                    replay_rng.shuffle(frame_slots)
                    replayed_particle_ids.append(frame_slots)
                replayed_particle_ids = np.concatenate(replayed_particle_ids)
            with np.load(paths.ground_truth, allow_pickle=False) as private:
                self.assertEqual(str(private["format_version"]), "1.3")
                self.assertTrue(required_private.issubset(private.files))
                self.assertEqual(private["particle_chromosome_ids"].shape, (24,))
                self.assertFalse(
                    np.array_equal(
                        replayed_particle_ids,
                        private["detection_particle_ids"],
                    )
                )
                self.assertEqual(
                    private["chromosome_centers_um"].shape, (4, 4, 2)
                )
                self.assertEqual(
                    private["chromosome_orientations_rad"].shape, (4, 4)
                )
                self.assertEqual(
                    private["chromosome_ellipse_semi_axes_um"].shape,
                    (4, 2),
                )
                self.assertEqual(
                    private["chromosome_initialization_bounds_um"].shape,
                    (4, 2, 2),
                )

    def test_animation_accepts_multiple_chromosomes(self) -> None:
        result = simulate(
            SimulationConfig(n_particles=24, n_frames=3, random_seed=25),
            create_multi_chromosome_bound_dynamic_2d(n_chromosomes=4),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chromosomes.gif"
            save_animation(result, output, dpi=30)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
