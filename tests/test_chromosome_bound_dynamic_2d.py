from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from srm_sim import (
    CHROMOSOME_BOUND_DYNAMIC_2D_ID,
    RigidPose2D,
    SimulationConfig,
    create_chromosome_bound_dynamic_2d,
    save_animation,
    save_output_bundle,
    simulate,
)
from srm_sim.models import (
    BrownianMotion2D,
    DynamicChromosomeScene2D,
    GaussianLocalizationObservation,
    ReflectingEllipseBoundary,
)


class ReflectingEllipseBoundaryTests(unittest.TestCase):
    def test_uniform_initialization_and_reflection_stay_inside(self) -> None:
        boundary = ReflectingEllipseBoundary(
            semi_major_axis_um=2.0,
            semi_minor_axis_um=0.35,
        )
        initial = boundary.sample_initial_positions(
            10_000, np.random.default_rng(1)
        )
        scaled_radius_squared = (
            np.square(initial[:, 0] / 2.0)
            + np.square(initial[:, 1] / 0.35)
        )
        self.assertLessEqual(float(scaled_radius_squared.max()), 1.0)

        previous = np.array(
            [[0.0, 0.0], [1.9, 0.0], [0.0, 0.34], [0.5, 0.1]],
            dtype=np.float64,
        )
        proposed = np.array(
            [[9.0, 0.0], [2.3, 0.0], [0.0, 0.5], [8.0, 2.0]],
            dtype=np.float64,
        )
        previous_copy = previous.copy()
        proposed_copy = proposed.copy()
        reflected = boundary.apply_step(previous, proposed)
        reflected_radius_squared = (
            np.square(reflected[:, 0] / 2.0)
            + np.square(reflected[:, 1] / 0.35)
        )
        self.assertLessEqual(float(reflected_radius_squared.max()), 1.0 + 1e-10)
        np.testing.assert_array_equal(previous, previous_copy)
        np.testing.assert_array_equal(proposed, proposed_copy)
        np.testing.assert_allclose(reflected[0], [1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(reflected[1], [1.7, 0.0], atol=1e-10)
        np.testing.assert_allclose(reflected[2], [0.0, 0.2], atol=1e-10)

        tangent = boundary.apply_step(
            np.array([[2.0, 0.0]]),
            np.array([[2.0, 0.1]]),
        )
        tangent_radius_squared = (
            (tangent[0, 0] / 2.0) ** 2
            + (tangent[0, 1] / 0.35) ** 2
        )
        self.assertLessEqual(tangent_radius_squared, 1.0)

        near_boundary = np.array([[2.0 * np.sqrt(1.0 + 5e-13), 0.0]])
        stationary = boundary.apply_step(near_boundary, near_boundary)
        self.assertLessEqual(
            float((stationary[0, 0] / 2.0) ** 2),
            1.0,
        )

    def test_local_brownian_variance_away_from_boundary(self) -> None:
        diffusion = 0.005
        dt = 0.05
        positions = np.zeros((100_000, 2), dtype=np.float64)
        proposed = BrownianMotion2D(diffusion).step(
            positions, dt, np.random.default_rng(2)
        )
        reflected = ReflectingEllipseBoundary(100.0, 100.0).apply_step(
            positions, proposed
        )
        np.testing.assert_allclose(
            (reflected - positions).var(axis=0),
            2.0 * diffusion * dt,
            rtol=0.03,
        )


class GaussianLocalizationObservationTests(unittest.TestCase):
    def test_noise_is_visible_only_and_has_configured_variance(self) -> None:
        sigma = 0.03
        observation = GaussianLocalizationObservation(
            lower_bounds_um=(0.0, 0.0),
            upper_bounds_um=(10.0, 10.0),
            localization_sigma_um=sigma,
        )
        positions = np.full((100_001, 2), 5.0, dtype=np.float64)
        emitting = np.ones(100_001, dtype=np.bool_)
        emitting[-1] = False
        observed, visible = observation.observe(
            positions, emitting, np.random.default_rng(3)
        )
        self.assertTrue(visible[:-1].all())
        self.assertFalse(visible[-1])
        self.assertTrue(np.isnan(observed[-1]).all())
        noise = observed[:-1] - positions[:-1]
        np.testing.assert_allclose(noise.mean(axis=0), 0.0, atol=3e-4)
        np.testing.assert_allclose(noise.var(axis=0), sigma**2, rtol=0.02)

        exact = GaussianLocalizationObservation(localization_sigma_um=0.0)
        exact_observed, exact_visible = exact.observe(
            positions[:10], emitting[:10], np.random.default_rng(4)
        )
        self.assertTrue(exact_visible.all())
        np.testing.assert_array_equal(exact_observed, positions[:10])


class ChromosomeScenarioTests(unittest.TestCase):
    def test_ou_update_is_stable_near_brownian_limit(self) -> None:
        scene = DynamicChromosomeScene2D(
            chromosome_diffusion_coefficient_um2_s=1.0,
            chromosome_relaxation_rate_s=1e-16,
            rotational_diffusion_rad2_s=0.0,
            equilibrium_center_um=(0.0, 0.0),
            initial_center_um=(0.0, 0.0),
        )
        expected_rng = np.random.default_rng(11)
        expected_center = (
            np.sqrt(2.0 * 1.0 * 0.05)
            * expected_rng.normal(0.0, 1.0, size=2)
        )
        pose = scene.step_pose(
            RigidPose2D((0.0, 0.0), 0.0),
            0.05,
            np.random.default_rng(11),
        )
        np.testing.assert_allclose(pose.center_um, expected_center, rtol=1e-14)

    def test_coupled_motion_confinement_and_reproducibility(self) -> None:
        scenario = create_chromosome_bound_dynamic_2d()
        config = SimulationConfig(
            n_particles=200,
            n_frames=80,
            frame_interval_s=0.05,
            random_seed=7,
        )
        result = simulate(config, scenario)
        repeated = simulate(config, scenario)

        self.assertEqual(scenario.scenario_id, CHROMOSOME_BOUND_DYNAMIC_2D_ID)
        self.assertIsNotNone(result.local_positions_um)
        self.assertIsNotNone(result.scene_centers_um)
        self.assertIsNotNone(result.scene_orientations_rad)
        local = result.local_positions_um
        centers = result.scene_centers_um
        orientations = result.scene_orientations_rad
        assert local is not None and centers is not None and orientations is not None

        scaled_radius_squared = (
            np.square(local[..., 0] / 2.0)
            + np.square(local[..., 1] / 0.35)
        )
        self.assertLessEqual(float(scaled_radius_squared.max()), 1.0 + 1e-10)

        cosine = np.cos(orientations)[:, None]
        sine = np.sin(orientations)[:, None]
        reconstructed = np.empty_like(result.positions_um)
        reconstructed[..., 0] = (
            centers[:, None, 0]
            + cosine * local[..., 0]
            - sine * local[..., 1]
        )
        reconstructed[..., 1] = (
            centers[:, None, 1]
            + sine * local[..., 0]
            + cosine * local[..., 1]
        )
        np.testing.assert_allclose(reconstructed, result.positions_um, atol=1e-14)
        np.testing.assert_array_equal(result.positions_um, repeated.positions_um)
        np.testing.assert_array_equal(local, repeated.local_positions_um)
        np.testing.assert_array_equal(centers, repeated.scene_centers_um)
        np.testing.assert_array_equal(orientations, repeated.scene_orientations_rad)
        np.testing.assert_array_equal(result.states, repeated.states)
        np.testing.assert_array_equal(
            result.observed_positions_um,
            repeated.observed_positions_um,
        )

    def test_shared_translation_and_static_chromosome_controls(self) -> None:
        config = SimulationConfig(
            n_particles=30,
            n_frames=20,
            frame_interval_s=0.05,
            random_seed=8,
        )
        translating = create_chromosome_bound_dynamic_2d(
            protein_diffusion_coefficient_um2_s=0.0,
            chromosome_diffusion_coefficient_um2_s=0.001,
            rotational_diffusion_rad2_s=0.0,
            localization_sigma_um=0.0,
        )
        translated = simulate(config, translating)
        displacements = np.diff(translated.positions_um, axis=0)
        np.testing.assert_allclose(
            displacements,
            np.broadcast_to(displacements[:, :1, :], displacements.shape),
            atol=1e-14,
        )
        np.testing.assert_array_equal(
            translated.local_positions_um,
            np.broadcast_to(
                translated.local_positions_um[0],
                translated.local_positions_um.shape,
            ),
        )

        rotating_too = create_chromosome_bound_dynamic_2d(
            protein_diffusion_coefficient_um2_s=0.0,
            chromosome_diffusion_coefficient_um2_s=0.001,
            rotational_diffusion_rad2_s=0.001,
        )
        translated_and_rotated = simulate(config, rotating_too)
        np.testing.assert_array_equal(
            translated.scene_centers_um,
            translated_and_rotated.scene_centers_um,
        )

        rotation_only = create_chromosome_bound_dynamic_2d(
            protein_diffusion_coefficient_um2_s=0.0,
            chromosome_diffusion_coefficient_um2_s=0.0,
            rotational_diffusion_rad2_s=0.001,
        )
        rotated = simulate(config, rotation_only)
        np.testing.assert_array_equal(
            rotated.scene_orientations_rad,
            translated_and_rotated.scene_orientations_rad,
        )

        static = create_chromosome_bound_dynamic_2d(
            chromosome_diffusion_coefficient_um2_s=0.0,
            rotational_diffusion_rad2_s=0.0,
            initial_center_um=(4.0, 6.0),
            equilibrium_center_um=(5.0, 5.0),
        )
        static_result = simulate(config, static)
        np.testing.assert_array_equal(
            static_result.scene_centers_um,
            np.broadcast_to((4.0, 6.0), (config.n_frames, 2)),
        )
        np.testing.assert_array_equal(
            static_result.scene_orientations_rad,
            np.zeros(config.n_frames),
        )

    def test_public_export_is_tracking_safe_and_private_truth_is_complete(self) -> None:
        scenario = create_chromosome_bound_dynamic_2d(
            initial_on_fraction=1.0,
            k_on_s=0.0,
            k_off_s=0.0,
            k_bleach_s=0.0,
        )
        result = simulate(
            SimulationConfig(n_particles=40, n_frames=4, random_seed=9),
            scenario,
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
        expected_private_scene_keys = {
            "observed_positions_um",
            "visible",
            "chromosome_local_positions_um",
            "chromosome_center_um",
            "chromosome_orientation_rad",
            "chromosome_ellipse_semi_axes_um",
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = save_output_bundle(result, directory)
            with np.load(paths.observations, allow_pickle=False) as public:
                self.assertEqual(set(public.files), expected_public_keys)
                for forbidden in (
                    "particle_ids",
                    "detection_particle_ids",
                    "chromosome_center_um",
                    "chromosome_orientation_rad",
                    "chromosome_local_positions_um",
                ):
                    self.assertNotIn(forbidden, public.files)
                public_positions = public["positions_um"].copy()
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
                self.assertEqual(str(private["format_version"]), "1.2")
                self.assertTrue(expected_private_scene_keys.issubset(private.files))
                np.testing.assert_array_equal(
                    private["positions_um"], result.positions_um
                )
                np.testing.assert_array_equal(
                    private["observed_positions_um"],
                    result.observed_positions_um,
                )
                np.testing.assert_array_equal(
                    private["chromosome_ellipse_semi_axes_um"],
                    np.array([2.0, 0.35]),
                )
                lab_initialization_bounds = private[
                    "physical_initialization_bounds_um"
                ]
                self.assertTrue(
                    np.all(
                        private["positions_um"][0]
                        >= lab_initialization_bounds[:, 0]
                    )
                )
                self.assertTrue(
                    np.all(
                        private["positions_um"][0]
                        <= lab_initialization_bounds[:, 1]
                    )
                )
                np.testing.assert_array_equal(
                    private["local_initialization_bounds_um"],
                    np.array([[-2.0, 2.0], [-0.35, 0.35]]),
                )
                self.assertFalse(
                    np.array_equal(
                        replayed_particle_ids,
                        private["detection_particle_ids"],
                    )
                )
                self.assertFalse(
                    np.array_equal(
                        public_positions,
                        private["detection_true_positions_um"][:, :2],
                    )
                )

    def test_animation_accepts_chromosome_scenario(self) -> None:
        result = simulate(
            SimulationConfig(n_particles=20, n_frames=2, random_seed=10),
            create_chromosome_bound_dynamic_2d(),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "chromosome.gif"
            save_animation(result, output, dpi=30)
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
