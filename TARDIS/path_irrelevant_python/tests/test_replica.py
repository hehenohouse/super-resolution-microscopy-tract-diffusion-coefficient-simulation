from __future__ import annotations

import csv
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

from tardis_tau1 import Tau1Config
from tardis_tau1.core import (
    CellResult,
    aggregate_replica,
    analyze_replica,
    decompose_distribution,
)
from tardis_tau1.matio import V73BaseRecord, load_v73_base_records
from tardis_tau1.outputs import (
    _build_all_cell_intra_figure,
    _build_distribution_figure,
    _build_peak_normalized_all_cell_intra_figure,
    _peak_normalized_intra_curves,
    save_replica_result,
)

try:
    import h5py
except ImportError:
    h5py = None


class ReplicaAggregationTests(unittest.TestCase):
    @unittest.skipUnless(h5py is not None, "h5py is not installed")
    def test_base_records_are_enumerated_once_in_matlab_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.mat"
            self._write_fixture(
                path,
                [
                    "WT_Miro1_T1",
                    "WT_Miro1_T1_SpotsMaskFiltered",
                    "WT_Miro1_T1_TracksFiltered",
                    "WT_Miro1_T2",
                    "WT_Miro1_T2_SpotsMaskFiltered",
                    "WT_Miro1_T2_TracksFiltered",
                ],
            )
            records, discovered, failures = load_v73_base_records(
                path, frame_interval_s=0.02, pixel_size_m=117e-9
            )
            self.assertEqual(discovered, ["WT_Miro1_T1", "WT_Miro1_T2"])
            self.assertEqual([record.name for record in records], discovered)
            self.assertEqual([record.record_index for record in records], [0, 3])
            self.assertEqual(failures, [])
            np.testing.assert_allclose(records[0].positions[0, 1:], [117e-9, 468e-9])

    @unittest.skipUnless(h5py is not None, "h5py is not installed")
    def test_duplicate_and_malformed_base_records_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "fixture.mat"
            self._write_fixture(
                path,
                ["duplicate", "duplicate", "missing_y"],
                missing_y={2},
            )
            records, discovered, failures = load_v73_base_records(
                path, frame_interval_s=0.02, pixel_size_m=117e-9
            )
            self.assertEqual(records, [])
            self.assertEqual(discovered, ["duplicate", "duplicate", "missing_y"])
            self.assertEqual(len(failures), 3)
            self.assertEqual(sum("occurs more than once" in row["error"] for row in failures), 2)
            self.assertIn("missing fields: y_data", failures[2]["error"])

    @mock.patch("tardis_tau1.core.collect_environment_provenance")
    @mock.patch("tardis_tau1.core.fingerprint_file")
    @mock.patch("tardis_tau1.core.load_v73_base_records")
    def test_replica_collects_shared_provenance_once(
        self,
        load_mock: mock.Mock,
        fingerprint_mock: mock.Mock,
        environment_mock: mock.Mock,
    ) -> None:
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 2.0, 0.0],
            ]
        )
        records = [
            V73BaseRecord("cell_A", 0, positions, {}),
            V73BaseRecord("cell_B", 1, positions, {}),
        ]
        load_mock.return_value = (records, ["cell_A", "cell_B"], [])
        fingerprint_mock.return_value = {"sha256": "f" * 64, "file_bytes": 7}
        environment_mock.return_value = {
            "config": {},
            "package": {"name": "tardis-tau1", "version": None},
            "python": {"version": "test", "implementation": "test"},
            "git": {"available": False, "commit": None, "dirty": None},
        }
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "replica.mat"
            input_path.write_bytes(b"fixture")
            result, cells = analyze_replica(
                input_path,
                condition="WT",
                target="Miro1",
                config=Tau1Config(
                    max_distance_m=3.0,
                    n_bins=3,
                    min_distance_background_fit=0.0,
                    save_plots=False,
                ),
            )
        fingerprint_mock.assert_called_once()
        environment_mock.assert_called_once()
        self.assertEqual(result.counts["total_candidate_pairs"], 8)
        self.assertEqual(result.source["file_sha256"], "f" * 64)
        self.assertTrue(np.isfinite(result.provenance["execution"]["analysis_runtime_seconds"]))
        self.assertEqual(len(cells), 2)
        self.assertTrue(all(cell.source["file_sha256"] == "f" * 64 for cell in cells))
        self.assertTrue(all(cell.provenance["input"]["sha256"] == "f" * 64 for cell in cells))

    def test_beta_is_clipped_without_clipping_negative_intra(self) -> None:
        config = Tau1Config(n_bins=4, min_distance_background_fit=0.5, save_plots=False)
        result = decompose_distribution(
            np.array([0.0, 0.0, 0.5, 0.5]),
            np.array([0.0, 0.0, 0.25, 0.25]),
            config,
        )
        self.assertEqual(result["beta_raw"], 2.0)
        self.assertEqual(result["beta"], 1.0)
        self.assertTrue(result["beta_was_clipped"])
        np.testing.assert_allclose(
            result["inter_contribution"] + result["intra_contribution"],
            [0.0, 0.0, 0.5, 0.5],
        )

    def test_cell_balanced_uses_mean_of_contributions(self) -> None:
        first = self._cell(
            "A", [8, 2], [3, 1], beta=0.2, background=[0.9, 0.1]
        )
        second = self._cell(
            "B", [2, 8], [1, 7], beta=0.8, background=[0.1, 0.9]
        )
        result = aggregate_replica([first, second], Tau1Config(n_bins=2, save_plots=False))
        self.assertEqual(set(result.ids), {"condition", "target", "replicate"})
        expected = np.mean(
            [first.distributions["inter_contribution"], second.distributions["inter_contribution"]],
            axis=0,
        )
        shortcut = (
            np.mean([0.2, 0.8])
            * np.mean([[0.9, 0.1], [0.1, 0.9]], axis=0)
        )
        np.testing.assert_allclose(result.distributions["inter_contribution"], expected)
        self.assertFalse(np.allclose(expected, shortcut))
        self.assertAlmostEqual(result.distributions["beta"], 0.5)
        expected_shape = np.mean(
            [
                first.distributions["intra_conditional"],
                second.distributions["intra_conditional"],
            ],
            axis=0,
        )
        old_contribution_derived_shape = (
            result.distributions["intra_contribution"]
            / (1.0 - result.distributions["beta"])
        )
        np.testing.assert_allclose(
            result.distributions["average_intra_conditional_shape"],
            expected_shape,
        )
        self.assertFalse(np.allclose(expected_shape, old_contribution_derived_shape))
        self.assertAlmostEqual(float(expected_shape.sum()), 1.0)
        np.testing.assert_allclose(
            result.distributions["total"],
            result.distributions["inter_contribution"]
            + result.distributions["intra_contribution"],
        )

    def test_replica_rejects_zero_intra_mass_before_shape_averaging(self) -> None:
        cell = self._cell(
            "A", [8, 2], [8, 2], beta=1.0, background=[0.8, 0.2]
        )
        with self.assertRaisesRegex(ValueError, "signed Intra mass"):
            aggregate_replica([cell], Tau1Config(n_bins=2, save_plots=False))

    def test_replica_preserves_raw_pair_count_totals(self) -> None:
        first = self._cell(
            "A", [90, 10], [10, 10], beta=0.2, background=[0.5, 0.5]
        )
        second = self._cell(
            "B", [1, 9], [0, 80], beta=0.8, background=[0.0, 1.0]
        )
        result = aggregate_replica(
            [first, second], Tau1Config(n_bins=2, save_plots=False)
        )
        self.assertEqual(result.counts["total_retained_pairs"], 110)
        self.assertEqual(result.counts["inter_retained_pairs"], 100)
        self.assertEqual(
            sum(row["total_retained_pairs"] for row in result.counts["per_cell"]),
            result.counts["total_retained_pairs"],
        )
        self.assertEqual(
            sum(row["inter_retained_pairs"] for row in result.counts["per_cell"]),
            result.counts["inter_retained_pairs"],
        )
        self.assertLessEqual(result.qc["metrics"]["reconstruction_error"], 1e-12)

    def test_replica_output_is_cell_balanced_only_and_saves_cell_intra(self) -> None:
        cells = [
            self._cell("A", [8, 2], [3, 1], beta=0.2, background=[0.9, 0.1]),
            self._cell("B", [2, 8], [1, 7], beta=0.8, background=[0.1, 0.9]),
        ]
        config = Tau1Config(n_bins=2, save_plots=True)
        result = aggregate_replica(cells, config)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for obsolete in (
                "cell_balanced_distributions.csv",
                "pair_weighted_distributions.csv",
                "mode_comparison.json",
                "aggregation_comparison.png",
            ):
                (output / obsolete).write_text("obsolete", encoding="utf-8")
            save_replica_result(result, cells, output, config)
            with np.load(output / "replica_result.npz") as arrays:
                self.assertIn("total", arrays)
                self.assertIn("total_density_um_inv", arrays)
                self.assertIn("positive_intra_total_positive_mass", arrays)
                self.assertEqual(
                    arrays["schema_version"].item(),
                    "tau1-replica-cell-balanced-v5",
                )
                self.assertNotIn("pair_weighted_total", arrays)
                self.assertEqual(arrays["cell_ids"].tolist(), ["A", "B"])
                np.testing.assert_allclose(
                    arrays["cell_intra_contributions"],
                    [
                        cells[0].distributions["intra_contribution"],
                        cells[1].distributions["intra_contribution"],
                    ],
                )
                np.testing.assert_allclose(
                    arrays["cell_intra_contributions"].mean(axis=0),
                    arrays["intra_contribution"],
                )
                np.testing.assert_allclose(
                    arrays["cell_intra_conditional_shapes"].mean(axis=0),
                    arrays["average_intra_conditional_shape"],
                )
                self.assertAlmostEqual(
                    float(arrays["average_intra_conditional_shape"].sum()), 1.0
                )
            payload = json.loads((output / "replica_result.json").read_text())
            self.assertIn("distributions", payload)
            self.assertNotIn("pair_weighted", payload)
            self.assertNotIn("comparison", payload)
            self.assertTrue((output / "replica_distributions.csv").is_file())
            self.assertTrue((output / "replica_summary.csv").is_file())
            self.assertTrue((output / "replica_distribution_full_range.png").is_file())
            self.assertTrue((output / "replica_distribution_zoom_0p5um.png").is_file())
            self.assertTrue((output / "all_cell_intra_full_range.png").is_file())
            self.assertTrue((output / "all_cell_intra_zoom_0p5um.png").is_file())
            self.assertTrue((output / "all_cell_intra_peak_normalized_full_range.png").is_file())
            self.assertTrue((output / "all_cell_intra_peak_normalized_zoom_0p5um.png").is_file())
            self.assertTrue(
                (output / "cell_peak_normalized_shape_summary.csv").is_file()
            )
            for obsolete in (
                "cell_balanced_distributions.csv",
                "pair_weighted_distributions.csv",
                "mode_comparison.json",
                "aggregation_comparison.png",
            ):
                self.assertFalse((output / obsolete).exists())
            with (output / "cell_summary.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("peak_density_um_inv", rows[0])
            self.assertTrue((output / "cells" / "A" / "cell_result.npz").is_file())

    def test_plots_have_full_and_zoom_density_panels(self) -> None:
        cells = [
            self._cell("A", [8, 2], [3, 1], beta=0.2, background=[0.9, 0.1]),
            self._cell("B", [2, 8], [1, 7], beta=0.8, background=[0.1, 0.9]),
        ]
        result = aggregate_replica(cells, Tau1Config(n_bins=2, save_plots=False))
        fig, axes = _build_distribution_figure(result, "fixture")
        self.assertEqual(len(axes), 2)
        self.assertEqual(axes[0].get_ylabel(), "Probability density (μm⁻¹)")
        self.assertAlmostEqual(axes[0].get_xlim()[0], 0.0)
        self.assertAlmostEqual(axes[1].get_xlim()[1], 0.5)
        np.testing.assert_allclose(
            axes[0].lines[2].get_ydata(),
            result.distributions["intra_contribution_density_um_inv"],
        )
        self.assertEqual(len(axes[1].lines), 2)  # Intra plus the zero reference line.
        np.testing.assert_allclose(
            axes[1].lines[0].get_ydata(),
            result.distributions["intra_contribution_density_um_inv"],
        )
        intra_fig, intra_axes = _build_all_cell_intra_figure(result, cells)
        self.assertEqual(len(intra_axes), 2)
        self.assertEqual(
            intra_axes[0].get_ylabel(), "Intra contribution density (μm⁻¹)"
        )
        normalized, peaks = _peak_normalized_intra_curves(cells)
        self.assertTrue(np.all(peaks > 0))
        np.testing.assert_allclose(normalized.max(axis=1), 1.0)
        normalized_fig, normalized_axes = _build_peak_normalized_all_cell_intra_figure(
            result, cells
        )
        self.assertEqual(len(normalized_axes), 2)
        self.assertEqual(
            normalized_axes[0].get_ylabel(), "Intra density / cell positive peak"
        )
        import matplotlib.pyplot as plt

        plt.close(fig)
        plt.close(intra_fig)
        plt.close(normalized_fig)

    @staticmethod
    def _cell(
        cell_id: str,
        total_counts: list[int],
        background_counts: list[int],
        *,
        beta: float,
        background: list[float],
    ) -> CellResult:
        total_histogram = np.asarray(total_counts, dtype=np.int64)
        inter_histogram = np.asarray(background_counts, dtype=np.int64)
        total = total_histogram / total_histogram.sum()
        background_array = np.asarray(background, dtype=float)
        inter = beta * background_array
        intra = total - inter
        return CellResult(
            ids={"condition": "WT", "target": "Miro1", "replicate": "R1", "cell": cell_id},
            source={},
            metadata={},
            parameters={
                **Tau1Config(n_bins=2, save_plots=False).to_dict(),
                "edges_m": np.array([0.0, 1.0, 2.0]),
                "bin_mids_m": np.array([0.5, 1.5]),
            },
            counts={
                "localizations": 10,
                "frames": 2,
                "total_candidate_pairs": int(total_histogram.sum()),
                "total_retained_pairs": int(total_histogram.sum()),
                "total_histogram": total_histogram,
                "inter_candidate_pairs": int(inter_histogram.sum()),
                "inter_retained_pairs": int(inter_histogram.sum()),
                "inter_histogram": inter_histogram,
            },
            distributions={
                "total": total,
                "inter_conditional": background_array,
                "inter_contribution": inter,
                "intra_contribution": intra,
                "intra_conditional": (
                    intra / (1 - beta)
                    if beta < 1.0
                    else np.full_like(intra, np.nan)
                ),
                "beta": beta,
                "beta_raw": beta,
                "beta_was_clipped": False,
            },
            qc={
                "pass": True,
                "status": "pass",
                "flags": [],
                "metrics": {
                    "negative_intra_bins": int(np.count_nonzero(intra < 0)),
                    "negative_intra_mass": float(-np.minimum(intra, 0).sum()),
                },
            },
            schema_version="tau1-cell-v2",
        )

    @staticmethod
    def _write_fixture(
        path: Path, names: list[str], *, missing_y: set[int] | None = None
    ) -> None:
        assert h5py is not None
        missing_y = missing_y or set()
        with h5py.File(path, "w") as handle:
            refs = handle.create_group("#refs#")
            references = []
            for index, name in enumerate(names):
                record = refs.create_group(f"record_{index}")
                record.create_dataset(
                    "name", data=np.asarray([[ord(char) for char in name]], dtype=np.uint16)
                )
                record.create_dataset("time", data=np.array([[4.00, 4.02, 4.04]]))
                record.create_dataset("x_data", data=np.array([[1.0, 2.0, 3.0]]))
                if index not in missing_y:
                    record.create_dataset("y_data", data=np.array([[4.0, 5.0, 6.0]]))
                metadata = record.create_group("tracksMetaData")
                metadata.create_dataset("frameInterval", data=np.array([[0.02]]))
                references.append(record.ref)
            data = handle.create_dataset("data", (1, len(references)), dtype=h5py.ref_dtype)
            data[0, :] = references


if __name__ == "__main__":
    unittest.main()
