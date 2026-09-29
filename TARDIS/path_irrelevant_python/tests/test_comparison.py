from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_xinran_all_groups import discover_input_groups
from tardis_tau1.comparison import (
    _conditional_record,
    _peak_normalized_record,
    _shape_summary,
    save_condition_target_comparison,
    save_target_condition_comparison,
)


class ComparisonTests(unittest.TestCase):
    def test_signed_net_mass_normalization_preserves_negative_bins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            group_dir = Path(temporary) / "group"
            self._write_group(
                group_dir.parent,
                np.array([0.0, 0.1e-6, 0.2e-6, 0.3e-6]),
                np.array([[2.0, -1.0, 1.0], [4.0, -2.0, 2.0]]),
            )
            record = _conditional_record(group_dir)

        widths_um = np.full(3, 0.1)
        np.testing.assert_allclose(record["mean"], [10.0, -5.0, 5.0])
        self.assertAlmostEqual(float(np.sum(record["mean"] * widths_um)), 1.0)
        self.assertLess(record["mean"][1], 0.0)
        # Dividing by positive mass instead would produce a first bin of 2 / 0.3.
        self.assertNotAlmostEqual(record["mean"][0], 2.0 / 0.3)

    def test_peak_normalization_occurs_before_replica_averaging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            group_dir = Path(temporary) / "group"
            self._write_group(
                group_dir.parent,
                np.array([0.0, 0.1e-6, 0.2e-6, 0.3e-6]),
                np.array([[2.0, 1.0, -1.0], [4.0, 0.0, -2.0]]),
            )
            record = _peak_normalized_record(group_dir)

        np.testing.assert_allclose(record["mean"], [1.0, 0.25, -0.5])
        np.testing.assert_allclose(
            record["sd"], np.std([[1.0, 0.5, -0.5], [1.0, 0.0, -0.5]], axis=0, ddof=1)
        )

    def test_invalid_signed_net_mass_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            group_dir = Path(temporary) / "group"
            self._write_group(
                group_dir.parent,
                np.array([0.0, 0.1e-6, 0.2e-6, 0.3e-6]),
                np.array([[1.0, -1.0, 0.0]]),
            )
            with self.assertRaisesRegex(ValueError, "Signed net Intra mass"):
                _conditional_record(group_dir)

    def test_comparison_rejects_mismatched_distance_bins(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wt = root / "WT"
            mdk = root / "MDK"
            self._write_group(
                wt,
                np.array([0.0, 0.1e-6, 0.2e-6, 0.3e-6]),
                np.array([[2.0, -1.0, 1.0]]),
            )
            self._write_group(
                mdk,
                np.array([0.0, 0.1e-6, 0.25e-6, 0.3e-6]),
                np.array([[2.0, -1.0, 1.0]]),
            )
            with self.assertRaisesRegex(ValueError, "distance bins do not match"):
                save_target_condition_comparison(
                    "Miro1", {"WT": wt, "MDK": mdk}, root / "comparison"
                )

    def test_condition_and_target_comparisons_write_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            edges = np.array([0.0, 0.1e-6, 0.2e-6, 0.3e-6])
            self._write_group(first, edges, np.array([[2.0, -1.0, 1.0], [3.0, -1.0, 2.0]]))
            self._write_group(second, edges, np.array([[4.0, -2.0, 2.0], [2.0, -0.5, 1.5]]))

            condition_output = root / "condition_output"
            save_target_condition_comparison(
                "Miro1", {"MDK": second, "WT": first}, condition_output
            )
            target_output = root / "target_output"
            save_condition_target_comparison(
                "WT", {"Miro2": second, "Miro1": first}, target_output
            )

            for name in (
                "intra_contribution_by_distance.csv",
                "intra_shape_summary.csv",
                "intra_contribution_full_range.png",
                "intra_contribution_zoom_0p5um.png",
                "intra_area_normalized_by_distance.csv",
                "intra_area_normalized_shape_summary.csv",
                "intra_area_normalized_full_range.png",
                "intra_area_normalized_zoom_0p5um.png",
                "intra_peak_normalized_by_distance.csv",
                "intra_peak_normalized_zoom_0p5um.png",
            ):
                self.assertTrue((condition_output / name).is_file(), name)
            with (condition_output / "intra_contribution_by_distance.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                fields = next(csv.reader(handle))
            self.assertLess(
                fields.index("WT_intra_mean_density_um_inv"),
                fields.index("MDK_intra_mean_density_um_inv"),
            )
            with (
                target_output / "targets_intra_contribution_by_distance.csv"
            ).open(newline="", encoding="utf-8") as handle:
                fields = next(csv.reader(handle))
            self.assertLess(
                fields.index("Miro2_intra_mean_density_um_inv"),
                fields.index("Miro1_intra_mean_density_um_inv"),
            )

    def test_shape_summary_interpolates_median_within_nonuniform_bin(self) -> None:
        edges = np.array([0.0, 0.1e-6, 0.4e-6])
        density = np.array([6.0, 4.0 / 3.0])

        summary = _shape_summary(edges, density)

        # Positive masses are 0.6 and 0.4, so the median is 5/6 through bin 1.
        self.assertAlmostEqual(summary["median_nm"], 100.0 * (0.5 / 0.6))
        self.assertAlmostEqual(summary["mode_nm"], 50.0)

    def test_input_group_discovery_is_filtered_and_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wt = root / "Miro1" / "Miro1_WT"
            wt.mkdir(parents=True)
            (wt / "b.mat").write_bytes(b"")
            (wt / "a.mat").write_bytes(b"")
            (wt / "notes.txt").write_text("ignored", encoding="utf-8")
            (root / "Miro1" / "Miro1_MDK").mkdir()
            (root / "Miro1" / "unrelated").mkdir()
            (root / "empty_target" / "unrelated").mkdir(parents=True)

            groups = discover_input_groups(root)

        self.assertEqual(list(groups), ["Miro1"])
        self.assertEqual(list(groups["Miro1"]), ["MDK", "WT"])
        self.assertEqual(
            [path.name for path in groups["Miro1"]["WT"]], ["a.mat", "b.mat"]
        )
        self.assertEqual(groups["Miro1"]["MDK"], [])

    @staticmethod
    def _write_group(parent: Path, edges: np.ndarray, raw: np.ndarray) -> None:
        group = parent / "group"
        group.mkdir(parents=True)
        mids = (edges[:-1] + edges[1:]) / 2
        mean = raw.mean(axis=0)
        sd = raw.std(axis=0, ddof=1) if raw.shape[0] > 1 else np.full(raw.shape[1], np.nan)
        np.savez_compressed(
            group / "group_result.npz",
            edges_m=edges,
            bin_mids_m=mids,
            intra_contribution_replica_density_um_inv=raw,
        )
        with (group / "group_mean_distributions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "distance_left_m",
                    "distance_right_m",
                    "distance_mid_m",
                    "intra_contribution_mean_density_um_inv",
                    "intra_contribution_sd_density_um_inv",
                ]
            )
            writer.writerows(zip(edges[:-1], edges[1:], mids, mean, sd))
        with (group / "group_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "metric",
                    "n_replicas",
                    "mean",
                    "sd",
                    "sem",
                    "mean_minus_sd",
                    "mean_plus_sd",
                ]
            )
            for metric, value in (
                ("peak_distance_um", 0.05),
                ("positive_weighted_mean_distance_um", 0.10),
                ("positive_weighted_median_distance_um", 0.08),
            ):
                writer.writerow([metric, raw.shape[0], value, 0.01, 0.01, value - 0.01, value + 0.01])


if __name__ == "__main__":
    unittest.main()
