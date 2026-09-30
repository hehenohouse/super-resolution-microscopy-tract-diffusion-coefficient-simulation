"""Content-addressed resume support for saved replica analyses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .config import Tau1Config
from .core import ReplicaResult
from .outputs import load_replica_result
from .provenance import fingerprint_file

CACHE_SCHEMA = "tau1-replica-cache-v1"
REPLICA_SCHEMA = "tau1-replica-cell-balanced-v5"


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def build_replica_identity(
    *,
    fingerprint: dict[str, Any],
    condition: str,
    target: str,
    replica: str,
    variable_name: str,
    frame_interval_s: float,
    pixel_size_m: float,
    config: Tau1Config,
) -> dict[str, Any]:
    return {
        "input": {
            "sha256": str(fingerprint["sha256"]).lower(),
            "file_bytes": int(fingerprint["file_bytes"]),
        },
        "ids": {"condition": condition, "target": target, "replicate": replica},
        "calibration": {
            "position_variable": variable_name,
            "frame_interval_s": float(frame_interval_s),
            "pixel_size_m": float(pixel_size_m),
        },
        "analysis_config": config.analysis_identity(),
        "replica_schema_version": REPLICA_SCHEMA,
    }


def identity_digest(identity: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()


def _required_artifacts(result: ReplicaResult, save_plots: bool) -> list[str]:
    names = [
        "replica_result.json",
        "replica_result.npz",
        "replica_distributions.csv",
        "cell_summary.csv",
        "replica_summary.csv",
        "cell_peak_normalized_shape_summary.csv",
    ]
    for cell_id in result.cells.get("included_cell_ids", []):
        safe = "".join(
            character if character.isalnum() or character in "_.-" else "_"
            for character in str(cell_id)
        ) or "UNASSIGNED"
        names.extend(
            [f"cells/{safe}/cell_result.json", f"cells/{safe}/cell_result.npz"]
        )
    if save_plots:
        for stem in (
            "replica_distribution",
            "intra_area_normalized",
            "all_cell_intra",
            "all_cell_intra_peak_normalized",
        ):
            names.extend(
                [f"{stem}_full_range.png", f"{stem}_zoom_0p5um.png"]
            )
    return names


def _result_matches_identity(
    result: ReplicaResult, identity: dict[str, Any]
) -> bool:
    source = result.source
    provenance = result.provenance
    calibration = identity["calibration"]
    stored_config = provenance.get("config")
    return (
        result.schema_version == identity["replica_schema_version"]
        and all(
            result.ids.get(key) == value for key, value in identity["ids"].items()
        )
        and str(source.get("file_sha256", "")).lower()
        == identity["input"]["sha256"]
        and int(source.get("file_bytes", -1)) == identity["input"]["file_bytes"]
        and source.get("position_variable") == calibration["position_variable"]
        and source.get("frame_interval_s") == calibration["frame_interval_s"]
        and source.get("pixel_size_m") == calibration["pixel_size_m"]
        and isinstance(stored_config, dict)
        and {
            key: stored_config.get(key) for key in identity["analysis_config"]
        }
        == identity["analysis_config"]
    )


def load_cached_replica(
    output_dir: str | Path,
    identity: dict[str, Any],
    *,
    save_plots: bool,
) -> ReplicaResult | None:
    """Return a strictly valid cached replica, otherwise return None."""
    output = Path(output_dir)
    manifest_path = output / "replica_cache.json"
    try:
        result = load_replica_result(output)
        if not _result_matches_identity(result, identity):
            return None
        required = _required_artifacts(result, save_plots)
        if not all((output / name).is_file() for name in required):
            return None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema_version") != CACHE_SCHEMA
                or manifest.get("status") != "complete"
                or manifest.get("identity") != identity
                or manifest.get("cache_key_sha256") != identity_digest(identity)
            ):
                return None
            checksums = manifest.get("result", {})
            for name in ("replica_result.json", "replica_result.npz"):
                key = f"{name}_sha256"
                if checksums.get(key) != fingerprint_file(output / name)["sha256"]:
                    return None
        else:
            write_cache_manifest(
                output, identity, result, save_plots=save_plots
            )
        return result
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def write_cache_manifest(
    output_dir: str | Path,
    identity: dict[str, Any],
    result: ReplicaResult,
    *,
    save_plots: bool,
) -> None:
    """Atomically mark a completely serialized replica as reusable."""
    output = Path(output_dir)
    artifacts = _required_artifacts(result, save_plots)
    missing = [name for name in artifacts if not (output / name).is_file()]
    if missing:
        raise ValueError(
            f"Cannot complete replica cache; missing artifacts: {missing}"
        )
    manifest = {
        "schema_version": CACHE_SCHEMA,
        "status": "complete",
        "cache_key_sha256": identity_digest(identity),
        "identity": identity,
        "result": {
            f"{name}_sha256": fingerprint_file(output / name)["sha256"]
            for name in ("replica_result.json", "replica_result.npz")
        },
        "included_cell_ids": list(result.cells.get("included_cell_ids", [])),
        "artifacts": artifacts,
    }
    temporary = output / "replica_cache.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, output / "replica_cache.json")
