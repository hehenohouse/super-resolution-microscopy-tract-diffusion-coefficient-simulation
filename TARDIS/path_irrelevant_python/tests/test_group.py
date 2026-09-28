from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

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

    @staticmethod
    def _replica(replica_id: str, total: list[float]) -> ReplicaResult:
        total_array = np.asarray(total, dtype=float)
        inter = total_array * 0.25
        intra = total_array - inter
        return ReplicaResult(
            ids={"condition": "MDK", "target": "Miro1", "replicate": replica_id},
            source={},
            cells={},
            parameters={"edges_m": np.array([0.0, 1e-6, 2e-6])},
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
            provenance={},
            schema_version="fixture",
        )


if __name__ == "__main__":
    unittest.main()
