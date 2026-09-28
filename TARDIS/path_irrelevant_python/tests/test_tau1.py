from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tardis_tau1 import Tau1Config, aggregate_genotype, analyze_cell, run_batch
from tardis_tau1.core import CellResult, pair_histogram
from tardis_tau1.matio import load_v73_cell_record

try:
    import h5py
except ImportError:
    h5py = None


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_MAT = REPO_ROOT / "testPos2.mat"


class Tau1Tests(unittest.TestCase):
    def test_ordered_pair_semantics(self) -> None:
        frames = np.array([1, 2])
        positions = [
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0], [2.0, 0.0]]),
        ]
        config = Tau1Config(
            max_distance_m=3.0,
            n_bins=3,
            chunk_size=1,
            bootstrap_count=0,
            save_plots=False,
        )
        edges = np.arange(4.0)
        total = pair_histogram(frames, positions, edges, "total", config)
        inter = pair_histogram(frames, positions, edges, "inter", config)
        self.assertEqual(total.candidate_pairs, 4)
        np.testing.assert_array_equal(total.histogram, [1, 2, 1])
        self.assertEqual(inter.candidate_pairs, 8)
        self.assertEqual(inter.zero_distances, 4)
        self.assertEqual(inter.retained_pairs, 4)
        np.testing.assert_array_equal(inter.histogram, [0, 2, 2])

    def test_equal_cell_aggregation(self) -> None:
        config = Tau1Config(bootstrap_count=10, save_plots=False)
        first = self._fake_cell("A", "R1", [0.8, 0.2], [0.3, 0.2])
        second = self._fake_cell("B", "R2", [0.2, 0.8], [0.1, 0.4])
        result = aggregate_genotype([first, second], config)
        np.testing.assert_allclose(result.distributions["total"], [0.5, 0.5])
        np.testing.assert_allclose(
            result.distributions["inter_contribution"], [0.2, 0.3]
        )
        np.testing.assert_allclose(
            result.distributions["total"],
            result.distributions["inter_contribution"]
            + result.distributions["intra_contribution"],
        )

    def test_testpos2_regression(self) -> None:
        result = analyze_cell(
            TEST_MAT,
            config=Tau1Config(n_bins=100, bootstrap_count=0, save_plots=False),
        )
        self.assertEqual(result.counts["localizations"], 189_590)
        self.assertEqual(result.counts["frames"], 2_500)
        self.assertEqual(result.counts["total_candidate_pairs"], 16_370_405)
        self.assertEqual(result.counts["total_retained_pairs"], 520_113)
        self.assertEqual(result.counts["inter_candidate_pairs"], 16_477_456)
        self.assertEqual(result.counts["inter_zero_distances_removed"], 189_632)
        self.assertEqual(result.counts["inter_retained_pairs"], 425_388)
        self.assertEqual(result.counts["total_histogram"][0], 7_482)
        self.assertEqual(result.counts["inter_histogram"][0], 2)
        self.assertAlmostEqual(
            result.distributions["beta"], 0.8177098334831255, places=14
        )
        self.assertEqual(result.qc["metrics"]["negative_intra_bins"], 40)
        self.assertAlmostEqual(
            result.qc["metrics"]["minimum_intra_contribution"],
            -0.0006306613549036076,
            places=16,
        )
        self.assertLess(result.qc["metrics"]["reconstruction_error"], 1e-15)

    @unittest.skipUnless(h5py is not None, "h5py is not installed")
    def test_v73_base_record_si_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.mat"
            self._write_v73_fixture(path)
            positions, metadata = load_v73_cell_record(
                path,
                "WT_Miro1_T1",
                frame_interval_s=0.02,
                pixel_size_m=117e-9,
            )
            np.testing.assert_array_equal(positions[:, 0], [200, 201, 202])
            np.testing.assert_allclose(
                positions[:, 1:],
                np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]) * 117e-9,
            )
            self.assertEqual(metadata["record_variant"], "base")
            self.assertEqual(metadata["input_coordinate_unit"], "pixel")
            self.assertEqual(metadata["analysis_coordinate_unit"], "m")
            self.assertAlmostEqual(metadata["frame_interval_s"], 0.02)

    @unittest.skipUnless(h5py is not None, "h5py is not installed")
    def test_v73_rejects_filtered_record_selector(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.mat"
            self._write_v73_fixture(path)
            with self.assertRaisesRegex(ValueError, "unfiltered Base record"):
                load_v73_cell_record(
                    path,
                    "WT_Miro1_T1_SpotsMaskFiltered",
                    frame_interval_s=0.02,
                    pixel_size_m=117e-9,
                )

    def test_batch_single_cell_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "condition_id",
                        "target_id",
                        "genotype_id",
                        "replicate_id",
                        "cell_id",
                        "input_file",
                        "position_variable",
                        "coordinate_unit",
                        "frame_time_s",
                        "enabled",
                    ]
                )
                writer.writerow(
                    ["WT", "Trak1", "G1", "R1", "C1", TEST_MAT, "pos", "m", "", "true"]
                )
            output = root / "output"
            result = run_batch(
                manifest,
                output,
                Tau1Config(bootstrap_count=0, save_plots=False),
            )
            self.assertEqual(result.successful_cells, 1)
            self.assertEqual(result.failed_cells, 0)
            self.assertEqual(len(result.genotype_results), 1)
            self.assertTrue(
                (output / "cells/WT/Trak1/G1/R1/C1/cell_result.npz").is_file()
            )
            self.assertTrue(
                (output / "genotypes/WT/Trak1/G1/genotype_result.npz").is_file()
            )
            csv_path = output / "cells/WT/Trak1/G1/R1/C1/distributions.csv"
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("distance_left_m", rows[0])
            self.assertIn("total_count", rows[0])
            self.assertIn("inter_background_count", rows[0])
            self.assertIn("inter_expected_count", rows[0])
            self.assertIn("intra_expected_count", rows[0])
            first_cell = result.cell_results[0]
            self.assertEqual(
                int(rows[0]["total_count"]), first_cell.counts["total_histogram"][0]
            )
            self.assertAlmostEqual(
                float(rows[0]["total_probability"]),
                first_cell.distributions["total"][0],
            )
            self.assertAlmostEqual(
                float(rows[0]["inter_expected_count"])
                + float(rows[0]["intra_expected_count"]),
                float(rows[0]["total_count"]),
                places=8,
            )

    @staticmethod
    def _write_v73_fixture(path: Path) -> None:
        assert h5py is not None
        with h5py.File(path, "w") as handle:
            refs = handle.create_group("#refs#")
            records = []
            for index, name in enumerate(
                [
                    "WT_Miro1_T1",
                    "WT_Miro1_T1_SpotsMaskFiltered",
                    "WT_Miro1_T1_TracksFiltered",
                ]
            ):
                record = refs.create_group(f"record_{index}")
                record.create_dataset(
                    "name",
                    data=np.asarray([[ord(char) for char in name]], dtype=np.uint16),
                )
                record.create_dataset("time", data=np.array([[4.00, 4.02, 4.04]]))
                record.create_dataset("x_data", data=np.array([[1.0, 2.0, 3.0]]))
                record.create_dataset("y_data", data=np.array([[4.0, 5.0, 6.0]]))
                metadata = record.create_group("tracksMetaData")
                metadata.create_dataset("frameInterval", data=np.array([[0.02]]))
                records.append(record.ref)
            data = handle.create_dataset("data", (1, len(records)), dtype=h5py.ref_dtype)
            data[0, :] = records

    @staticmethod
    def _fake_cell(
        cell_id: str, replicate: str, total: list[float], inter: list[float]
    ) -> CellResult:
        total_array = np.asarray(total)
        inter_array = np.asarray(inter)
        intra_array = total_array - inter_array
        beta = float(inter_array.sum())
        return CellResult(
            ids={
                "condition": "C",
                "target": "T",
                "genotype": "G",
                "replicate": replicate,
                "cell": cell_id,
            },
            source={},
            metadata={},
            parameters={"edges_m": np.array([0.0, 1.0, 2.0]), "bin_mids_m": np.array([0.5, 1.5])},
            counts={},
            distributions={
                "total": total_array,
                "inter_contribution": inter_array,
                "intra_contribution": intra_array,
                "inter_conditional": inter_array / beta,
                "intra_conditional": intra_array / (1 - beta),
                "beta": beta,
            },
            qc={"pass": True},
            schema_version="tau1-distributions-v1",
        )


if __name__ == "__main__":
    unittest.main()
