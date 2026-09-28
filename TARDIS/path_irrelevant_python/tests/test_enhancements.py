from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tardis_tau1 import Tau1Config
from tardis_tau1.core import add_distribution_densities, summarize_positive_intra
from tardis_tau1.outputs import _build_distribution_figure
from tardis_tau1.provenance import collect_environment_provenance, fingerprint_file


class DensityAndSummaryTests(unittest.TestCase):
    def test_density_uses_actual_nonuniform_bin_widths(self) -> None:
        edges = np.array([0.0, 0.1e-6, 0.4e-6])
        distributions = add_distribution_densities(
            {
                "total": np.array([0.25, 0.75]),
                "inter_contribution": np.array([0.1, 0.4]),
                "intra_contribution": np.array([0.15, 0.35]),
            },
            edges,
        )
        np.testing.assert_allclose(distributions["total_density_um_inv"], [2.5, 2.5])
        widths_um = np.diff(edges) * 1e6
        self.assertAlmostEqual(
            float(np.sum(distributions["total_density_um_inv"] * widths_um)), 1.0
        )
        self.assertAlmostEqual(
            float(
                np.sum(
                    distributions["inter_contribution_density_um_inv"] * widths_um
                )
            ),
            0.5,
        )
        self.assertAlmostEqual(
            float(
                np.sum(
                    distributions["intra_contribution_density_um_inv"] * widths_um
                )
            ),
            0.5,
        )

    def test_default_bin_width_is_ten_nanometres(self) -> None:
        config = Tau1Config()
        edges = np.linspace(0.0, config.max_distance_m, config.n_bins + 1)
        np.testing.assert_allclose(np.diff(edges) * 1e6, 0.01)

    def test_positive_intra_summary_preserves_signed_input(self) -> None:
        intra = np.array([0.25, -0.2, 0.75])
        original = intra.copy()
        edges = np.array([0.0, 0.2e-6, 0.4e-6, 0.6e-6])
        summary = summarize_positive_intra(intra, edges)
        np.testing.assert_array_equal(intra, original)
        self.assertEqual(summary["peak_bin_index_zero_based"], 2)
        self.assertAlmostEqual(summary["peak_distance_um"], 0.5)
        self.assertAlmostEqual(summary["peak_density_um_inv"], 3.75)
        self.assertAlmostEqual(summary["total_positive_mass"], 1.0)
        self.assertAlmostEqual(summary["positive_mass_0_300_nm"], 0.25)
        self.assertAlmostEqual(summary["positive_weighted_mean_distance_um"], 0.4)
        self.assertAlmostEqual(
            summary["positive_weighted_median_distance_um"],
            0.4 + (0.25 / 0.75) * 0.2,
        )

    def test_300_nm_window_uses_fractional_bin_overlap(self) -> None:
        summary = summarize_positive_intra(
            np.array([0.4, 0.6]), np.array([0.0, 0.4e-6, 0.8e-6])
        )
        self.assertAlmostEqual(summary["positive_mass_0_300_nm"], 0.3)

    def test_zero_positive_mass_returns_undefined_locations(self) -> None:
        summary = summarize_positive_intra(
            np.array([-0.2, 0.0]), np.array([0.0, 0.1e-6, 0.2e-6])
        )
        self.assertEqual(summary["peak_bin_index_zero_based"], -1)
        self.assertEqual(summary["total_positive_mass"], 0.0)
        self.assertTrue(math.isnan(summary["peak_distance_m"]))
        self.assertTrue(math.isnan(summary["positive_weighted_median_distance_m"]))


class ProvenanceTests(unittest.TestCase):
    def test_file_fingerprint_matches_known_sha256(self) -> None:
        payload = b"TARDIS provenance fixture\n"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.mat"
            path.write_bytes(payload)
            result = fingerprint_file(path, chunk_size=3)
        self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(result["file_bytes"], len(payload))

    @mock.patch("tardis_tau1.provenance.importlib.metadata.version", return_value="9.8.7")
    @mock.patch("tardis_tau1.provenance._git_provenance")
    def test_environment_provenance_records_config_and_versions(
        self, git_mock: mock.Mock, _version_mock: mock.Mock
    ) -> None:
        git_mock.return_value = {
            "available": True,
            "commit": "a" * 40,
            "dirty": True,
        }
        config = Tau1Config(n_bins=300)
        result = collect_environment_provenance(config)
        self.assertEqual(result["config"], config.to_dict())
        self.assertEqual(result["package"]["version"], "9.8.7")
        self.assertEqual(result["git"]["commit"], "a" * 40)
        self.assertTrue(result["git"]["dirty"])
        self.assertTrue(result["python"]["version"])


if __name__ == "__main__":
    unittest.main()
