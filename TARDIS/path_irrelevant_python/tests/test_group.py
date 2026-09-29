from __future__ import annotations

import csv
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tardis_tau1.config import Tau1Config
from tardis_tau1.core import ReplicaResult
from tardis_tau1.group import aggregate_group, save_group_result


class GroupAggregationTests(unittest.TestCase):
    def test_replicas_are_equally_weighted_with_standard_deviation_bands(self) -> None:
        group = aggregate_group(
            [self._replica("R1", [0.2, 0.8]), self._replica("R2", [0.4, 0.6]), self._replica("R3", [0.6, 0.4])]
        )
        total = group.distributions["total"]
        np.testing.assert_allclose(total["mean_density_um_inv"], [0.4, 0.6])
        np.testing.assert_allclose(total["sd_density_um_inv"], [0.2, 0.2])
        np.testing.assert_allclose(total["sem_density_um_inv"], [0.2 / np.sqrt(3)] * 2)
        np.testing.assert_allclose(
            total["mean_minus_sd_density_um_inv"], [0.2, 0.4]
        )
        np.testing.assert_allclose(
            total["mean_plus_sd_density_um_inv"], [0.6, 0.8]
        )
        np.testing.assert_allclose(
            total["mean_density_um_inv"],
            group.distributions["inter_contribution"]["mean_density_um_inv"]
            + group.distributions["intra_contribution"]["mean_density_um_inv"],
        )

    def test_group_writes_tables_and_arrays(self) -> None:
        result = aggregate_group([self._replica("R1", [0.2, 0.8]), self._replica("R2", [0.4, 0.6])])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_group_result(result, output, save_plots=False)
            self.assertTrue((output / "group_mean_distributions.csv").is_file())
            self.assertTrue((output / "group_summary.csv").is_file())
            self.assertTrue((output / "replica_summary.csv").is_file())
            self.assertTrue((output / "group_result.npz").is_file())
            with (output / "group_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("total_positive_mass", [row["metric"] for row in rows])
            self.assertTrue((output / "replica_metric_summary.csv").is_file())
            self.assertTrue((output / "group_mean_curve_summary.csv").is_file())
            self.assertEqual(result.schema_version, "tau1-group-replica-balanced-v2")
            self.assertEqual(
                [row["replica_id"] for row in result.provenance["replicas"]],
                ["R1", "R2"],
            )

    def test_group_rejects_incompatible_or_duplicate_replicas(self) -> None:
        first = self._replica("R1", [0.2, 0.8])
        duplicate_id = self._replica("R1", [0.4, 0.6])
        with self.assertRaisesRegex(ValueError, "Duplicate replica ID"):
            aggregate_group([first, duplicate_id])

        mismatched_config = self._replica("R2", [0.4, 0.6])
        mismatched_config.provenance["config"]["max_distance_m"] = 4e-6
        mismatched_config.parameters["max_distance_m"] = 4e-6
        with self.assertRaisesRegex(ValueError, "incompatible analysis config"):
            aggregate_group([first, mismatched_config])

        mismatched_calibration = self._replica("R2", [0.4, 0.6])
        mismatched_calibration.source["pixel_size_m"] = 120e-9
        with self.assertRaisesRegex(ValueError, "incompatible calibration"):
            aggregate_group([first, mismatched_calibration])

        duplicate_input = copy.deepcopy(first)
        duplicate_input.ids["replicate"] = "R2"
        with self.assertRaisesRegex(ValueError, "Duplicate biological-replica"):
            aggregate_group([first, duplicate_input])

    def test_group_mean_curve_metrics_are_separate_from_replica_metrics(self) -> None:
        first = self._replica("R1", [0.9, 0.1])
        second = self._replica("R2", [0.1, 0.9])
        group = aggregate_group([first, second])

        self.assertIn("positive_weighted_median_distance_um", group.group_mean_curve_summary)
        self.assertIn(
            "positive_weighted_median_distance_um",
            group.replica_metric_statistics,
        )
        self.assertIs(group.summary_statistics, group.replica_metric_statistics)
        self.assertNotEqual(
            group.replica_metric_statistics[
                "positive_weighted_median_distance_um"
            ]["mean"],
            group.group_mean_curve_summary[
                "positive_weighted_median_distance_um"
            ],
        )

    @staticmethod
    def _replica(replica_id: str, total: list[float]) -> ReplicaResult:
        total_array = np.asarray(total, dtype=float)
        inter = total_array * 0.25
        intra = total_array - inter
        digest = f"{int(replica_id[1:]):064x}"
        config = Tau1Config().to_dict()
        return ReplicaResult(
            ids={"condition": "MDK", "target": "Miro1", "replicate": replica_id},
            source={
                "input_file": f"{replica_id}.mat",
                "position_variable": "data",
                "file_bytes": 100,
                "file_sha256": digest,
                "frame_interval_s": 0.02,
                "pixel_size_m": 117e-9,
            },
            cells={},
            parameters={
                **config,
                "edges_m": np.array([0.0, 1e-6, 2e-6]),
                "bin_mids_m": np.array([0.5e-6, 1.5e-6]),
            },
            counts={"n_cells": 10},
            distributions={
                "total": total_array,
                "inter_contribution": inter,
                "intra_contribution": intra,
                "beta": 0.25,
            },
            summaries={
                "positive_intra": {
                    "total_positive_mass": float(intra.sum()),
                    "positive_mass_0_300_nm": 0.1,
                    "peak_distance_um": 0.5,
                    "peak_density_um_inv": 0.6,
                    "positive_weighted_mean_distance_um": 0.9,
                    "positive_weighted_median_distance_um": 0.8,
                }
            },
            qc={},
            provenance={
                "config": config,
                "input": {
                    "path": f"{replica_id}.mat",
                    "sha256": digest,
                    "file_bytes": 100,
                },
                "package": {},
                "python": {},
                "git": {},
            },
            schema_version="tau1-replica-cell-balanced-v4",
        )


if __name__ == "__main__":
    unittest.main()
