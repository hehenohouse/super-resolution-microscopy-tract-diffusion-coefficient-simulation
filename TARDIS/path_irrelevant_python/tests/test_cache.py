from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tardis_tau1.cache import (
    build_replica_identity,
    identity_digest,
    load_cached_replica,
    write_cache_manifest,
)
from tardis_tau1.config import Tau1Config
from tardis_tau1.outputs import load_replica_result


class ReplicaCacheTests(unittest.TestCase):
    def test_identity_is_stable_and_changes_with_scientific_inputs(self) -> None:
        config = Tau1Config(save_plots=False)
        arguments = {
            "fingerprint": {"sha256": "a" * 64, "file_bytes": 10},
            "condition": "WT",
            "target": "Miro1",
            "replica": "R1",
            "variable_name": "data",
            "frame_interval_s": 0.02,
            "pixel_size_m": 117e-9,
            "config": config,
        }
        first = build_replica_identity(**arguments)
        second = build_replica_identity(**dict(reversed(list(arguments.items()))))
        self.assertEqual(identity_digest(first), identity_digest(second))

        changed = dict(arguments)
        changed["config"] = Tau1Config(n_bins=200, save_plots=False)
        self.assertNotEqual(
            identity_digest(first),
            identity_digest(build_replica_identity(**changed)),
        )
        changed = dict(arguments)
        changed["pixel_size_m"] = 120e-9
        self.assertNotEqual(
            identity_digest(first),
            identity_digest(build_replica_identity(**changed)),
        )

    def test_valid_cache_loads_and_checksum_damage_invalidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            identity = self._write_saved_replica(output)
            result = load_replica_result(output)
            write_cache_manifest(output, identity, result, save_plots=False)

            cached = load_cached_replica(output, identity, save_plots=False)
            self.assertIsNotNone(cached)
            np.testing.assert_array_equal(
                cached.distributions["intra_contribution"], [0.15, 0.60]
            )

            with (output / "replica_result.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            self.assertIsNone(
                load_cached_replica(output, identity, save_plots=False)
            )

    def test_legacy_valid_result_is_adopted_with_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            identity = self._write_saved_replica(output)

            cached = load_cached_replica(output, identity, save_plots=False)

            self.assertIsNotNone(cached)
            self.assertTrue((output / "replica_cache.json").is_file())

    @staticmethod
    def _write_saved_replica(output: Path) -> dict[str, object]:
        output.mkdir(parents=True, exist_ok=True)
        config = Tau1Config(save_plots=False)
        digest = "a" * 64
        identity = build_replica_identity(
            fingerprint={"sha256": digest, "file_bytes": 10},
            condition="WT",
            target="Miro1",
            replica="R1",
            variable_name="data",
            frame_interval_s=0.02,
            pixel_size_m=117e-9,
            config=config,
        )
        edges = np.array([0.0, 1e-6, 2e-6])
        distributions = {
            "total": [0.2, 0.8],
            "inter_contribution": [0.05, 0.20],
            "intra_contribution": [0.15, 0.60],
            "inter_conditional": [0.2, 0.8],
            "intra_conditional": [0.2, 0.8],
            "average_inter_conditional_shape": [0.2, 0.8],
            "beta": 0.25,
            "aggregation": "equal_cell_mean",
        }
        metadata = {
            "ids": {"condition": "WT", "target": "Miro1", "replicate": "R1"},
            "source": {
                "input_file": "R1.mat",
                "position_variable": "data",
                "file_bytes": 10,
                "file_sha256": digest,
                "frame_interval_s": 0.02,
                "pixel_size_m": 117e-9,
            },
            "cells": {"included_cell_ids": [], "included_count": 0},
            "parameters": {
                **config.to_dict(),
                "edges_m": edges.tolist(),
                "bin_mids_m": [0.5e-6, 1.5e-6],
            },
            "counts": {"n_cells": 0},
            "distributions": distributions,
            "qc": {"pass": True},
            "schema_version": "tau1-replica-cell-balanced-v4",
            "summaries": {"positive_intra": {}},
            "provenance": {
                "config": config.to_dict(),
                "input": {"path": "R1.mat", "sha256": digest, "file_bytes": 10},
            },
        }
        (output / "replica_result.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        np.savez_compressed(
            output / "replica_result.npz",
            edges_m=edges,
            bin_mids_m=np.array([0.5e-6, 1.5e-6]),
            total=np.array(distributions["total"]),
            inter_contribution=np.array(distributions["inter_contribution"]),
            intra_contribution=np.array(distributions["intra_contribution"]),
            inter_conditional=np.array(distributions["inter_conditional"]),
            intra_conditional=np.array(distributions["intra_conditional"]),
            average_inter_conditional_shape=np.array(
                distributions["average_inter_conditional_shape"]
            ),
            beta=np.array(0.25),
            schema_version=np.array("tau1-replica-cell-balanced-v4"),
        )
        for name in (
            "replica_distributions.csv",
            "cell_summary.csv",
            "replica_summary.csv",
            "cell_peak_normalized_shape_summary.csv",
        ):
            (output / name).write_text("fixture\n", encoding="utf-8")
        return identity


if __name__ == "__main__":
    unittest.main()
