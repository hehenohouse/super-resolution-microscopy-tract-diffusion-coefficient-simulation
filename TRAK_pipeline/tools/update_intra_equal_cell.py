#!/usr/bin/env python3
"""Update only completed-cell Intra hierarchies with equal-cell area means.

No cell analysis, scalar metric, Excel, localization, mask or TARDIS entrypoint
is imported or called. Scientific aggregation and publication are explicit.
All controlled writes, temporary files and caches remain on the data drive.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

SOURCE_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = SOURCE_ROOT.parents[1] if SOURCE_ROOT.parent.name == "releases" else SOURCE_ROOT
COMPAT = SOURCE_ROOT / "src/trak_pipeline/compat"
METHOD = {
    "method_id": "hierarchy_summary",
    "method_version": "2",
    "area_order": "normalize_cells_then_equal_cell_mean",
    "condition_area_order": "equal_BR_mean_of_corrected_area_means",
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def configure(path):
    path = Path(path).resolve()
    config = read(path)
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported dataset configuration")
    root = (path.parent / config["data_root"]).resolve()
    control = (root / config.get("control_relative", "_batch_pipeline")).resolve()
    mount = PACKAGE.parent.resolve()
    if not mount.is_mount() or mount == Path("/") or not root.is_relative_to(mount):
        raise ValueError("The independent external data volume must be mounted")
    if not root.is_dir() or not control.is_dir():
        raise FileNotFoundError("External dataset/control directory is unavailable")
    summary = config.get("summary", {})
    for key, value in METHOD.items():
        if summary.get(key) != value:
            raise ValueError("Configuration must explicitly select corrected v2 rule: " + key)
    inventory_release = (path.parent / config["legacy_releases"]["windows"]).resolve()
    if not (inventory_release / "expected_inventory.json").is_file():
        raise FileNotFoundError("Read-only inventory metadata is absent: " + str(inventory_release))
    cache = PACKAGE / "runtime/intra_equal_cell_v2"
    for name in ("tmp", "matplotlib", "python", "xdg"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "TMPDIR": str(cache / "tmp"), "TMP": str(cache / "tmp"),
        "TEMP": str(cache / "tmp"), "MPLCONFIGDIR": str(cache / "matplotlib"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "PYTHONPYCACHEPREFIX": str(cache / "python"),
        "PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg",
    })
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(COMPAT))
    import aggregate_intra as storage
    import platform_runtime
    platform_runtime.assert_drive_root(root)
    storage.CONTROL, storage.DATA_ROOT, storage.RELEASE = control, root, inventory_release
    return path, config, root, control, inventory_release, storage


def science_parameters(config):
    return dict(METHOD,
        frames=2500, tau_frames=1, frame_interval_s=0.02,
        distance_edges_um=[i / 100 for i in range(301)],
        groups=["On", "Not-on"], views=["signed", "area", "peak"],
        cell_weighting="equal", condition_weighting="equal_BR",
        signed_area_min_abs=float(config["summary"]["signed_area_min_abs"]),
        require_both_groups=True, require_positive_peak=True,
        shared_view_cohort=True, SD_ddof=1, smoothing=False,
        BR_peak_order="mean_of_cell_positive_peak_normalized_curves",
        condition_peak_order="mean_of_BR_positive_peak_normalized_canonical_curves")


def snapshots(root, refs):
    result = {}
    for ref in refs:
        path = (root / ref["relative_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Protected source escapes dataset root")
        stat = path.stat()
        result[path.relative_to(root).as_posix()] = {
            "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        }
    return result


def verify_protected(root, before):
    after = snapshots(root, [{"relative_path": p} for p in before])
    if after != before:
        raise RuntimeError("A protected saved cell source or completion record changed")
    return after


def prior_state(control):
    path = control / "intra_summary_status.json"
    return read(path) if path.is_file() else {}


def prior_node(root, old):
    import intra_summary_core as core
    return core.portable_path(root, old["data"])


def verify_cohort(root, prior, nodes):
    old_nodes = {n["key"]: n for n in prior.get("nodes", [])}
    comparisons = []
    for node in nodes:
        old = old_nodes.get(node["key"])
        if not old:
            comparisons.append({"node": node["key"], "comparison": "no_previous_node"})
            continue
        path = prior_node(root, old) / "members_and_exclusions.json"
        if not path.is_file():
            raise FileNotFoundError("Previous cohort record is absent: " + str(path))
        previous = read(path)
        if previous.get("members", []) != node.get("members", []):
            raise ValueError("Shared scientific cohort changed unexpectedly: " + node["key"])
        comparisons.append({"node": node["key"], "comparison": "identical_members",
                            "members": len(node.get("members", []))})
    return comparisons


def curve_records(path, views=("signed", "peak")):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return {
            (r["spatial_group"], r["normalization"], r["series_kind"],
             r["series_id"], r["distance_left_um"]):
            (r["value"], r["SD"], r["SEM"], r["n"])
            for r in csv.DictReader(stream) if r["normalization"] in views
        }


def verify_unchanged_views(root, prior, records):
    old_nodes = {n["key"]: n for n in prior.get("nodes", [])}
    checked = []
    for record in records:
        old = old_nodes.get(record["key"])
        if not old:
            continue
        old_path = prior_node(root, old) / "curves_long.csv.gz"
        new_path = root / record["data_relative"] / "curves_long.csv.gz"
        if not old_path.is_file():
            raise FileNotFoundError("Previous full-precision hierarchy is absent: " + str(old_path))
        old_rows, new_rows = curve_records(old_path), curve_records(new_path)
        if old_rows != new_rows:
            raise ValueError("Signed or peak view changed unexpectedly: " + record["key"])
        checked.append({"node": record["key"], "identical_signed_and_peak_rows": len(new_rows)})
    return checked


def compute(config_path, config, root, control, release, storage):
    import intra_summary_core as core
    if str(core.SUMMARY_METHOD_VERSION) != "2":
        raise ValueError("Corrected v2 core is required")
    print("Reading committed cell Intra curves; no upstream scientific stage will run.", flush=True)
    science = science_parameters(config)
    definitions = [p for p in (control / "scientific_definition.json",
                               control / "windows/scientific_definition.json") if p.is_file()]
    fingerprints = {read(p)["fingerprint"] for p in definitions}
    if len(fingerprints) != 1:
        raise ValueError("Saved scientific definitions are absent or conflicting")
    built = core.build_summaries(control, release, {
        "molecule": config["molecule"],
        "scientific_fingerprint": fingerprints.pop(),
        **science,
    })
    nodes = built["nodes"]
    if len(nodes) != 13:
        raise ValueError("Expected 9 BR, 3 condition and 1 comparison node")
    old = prior_state(control)
    cohorts = verify_cohort(root, old, nodes)
    sources = {
        sid: {"result_digest": cell.get("result_digest"),
              "status": cell.get("status"), "eligible": cell.get("eligible"),
              "reasons": cell.get("reasons", [])}
        for sid, cell in sorted(built["cells"].items())
    }
    # Saved result digests bind curve values and validity, never program text.
    identity = {"method": science, "molecule": config["molecule"], "source_results": sources}
    summary_id = "v2_" + storage.digest(identity)[:12]
    refs = {ref["relative_path"]: ref for node in nodes for ref in node.get("source_refs", [])}
    for path in (*definitions, release / "expected_inventory.json",
                 control / "state/active_batch.json",
                 control / "windows/state/active_batch.json",
                 control / "windows/portable_completion_registry.json",
                 control / "windows/state/completed_cells_portable.json"):
        if path.is_file() and path.is_relative_to(root):
            refs.setdefault(path.relative_to(root).as_posix(),
                            {"relative_path": path.relative_to(root).as_posix()})
    before = snapshots(root, refs.values())
    records = []
    for node in nodes:
        node["molecule"] = config["molecule"]
        data = storage.guarded(Path(node["directory"]) / "analysis_results" / ("intra_summary_" + summary_id))
        data.mkdir(parents=True, exist_ok=True)
        storage.atomic_json(data / "COMPLETE.json", {
            "complete": False, "status": "UPDATING", "summary_id": summary_id,
            "started_utc": utc(),
        })
        storage.save_curves(node, data / "curves_long.csv.gz")
        storage.atomic_json(data / "members_and_exclusions.json", {
            "members": node.get("members", []), "exclusions": node.get("exclusions", []),
            "progress": node["progress"], "common_On_NotOn_and_all_views": True,
        })
        metadata = {
            "schema_version": 2, "summary_id": summary_id,
            "version": {"method": science}, "node_key": node["key"],
            "level": node["level"], "molecule": config["molecule"],
            "condition": node.get("condition"), "BR": node.get("BR"),
            "progress": node["progress"], "available": node.get("available"),
            "blocked": node.get("blocked"), "source_refs": node.get("source_refs", []),
            "signature_payload": node.get("signature_payload"),
            "series_metadata": {g: {v: {"n": p.get("n", 0),
                                 "series_ids": list(p.get("series", {}))}
                                for v, p in views.items()}
                                for g, views in node["groups"].items()},
            "edges_um": node["edges"], "distance_mid_um": node["x"],
            "scientific_aggregation": METHOD,
            "normalization_range_um": [0, 3], "final_renormalization": False,
            "uncertainty": "Between-BR sample SD; SEM saved; no time bootstrap or significance test.",
            "upstream_scientific_stages_run": [],
            "new_cells_started": 0,
        }
        storage.atomic_json(data / "metadata.json", metadata)
        artifacts = [data / name for name in
                     ("curves_long.csv.gz", "members_and_exclusions.json", "metadata.json")]
        signature = storage.digest({"scientific_identity": identity, "node_key": node["key"]})
        storage.atomic_json(data / "COMPLETE.json", {
            "complete": True, "signature": signature, "summary_id": summary_id,
            "completed_utc": utc(), "progress": node["progress"],
            "artifacts": [storage.descriptor(p) for p in artifacts],
            "completion_scope": "scientific_curves_only; publication tracked separately",
        })
        records.append({
            "key": node["key"], "level": node["level"], "status": "scientific_committed",
            "progress": node["progress"],
            "figures": str(Path(node["directory"]) / "Analysis_Figures_Intra"),
            "data": str(data), "data_relative": data.relative_to(root).as_posix(),
            "signature": signature,
        })
        print(config["molecule"] + " | saved " + node["key"], flush=True)
    unchanged = verify_unchanged_views(root, old, records)
    after = verify_protected(root, before)
    audit_paths = [Path(__file__), config_path] + [
        COMPAT / name for name in
        ("intra_summary_core.py", "intra_summary_plots.py",
         "aggregate_intra.py", "render_on_notoverlays.py")]
    report = {
        "schema_version": 2, "summary_id": summary_id,
        "created_utc": utc(), "molecule": config["molecule"],
        "method": science, "scientific_identity": identity,
        "audit_sources": [{"path": str(p.relative_to(PACKAGE)),
                          "sha256": storage.sha(p)} for p in audit_paths],
        "source_code_hashes_used_for_scientific_cache_identity": False,
        "discovery": built["discovery"], "nodes": records, "errors": [],
        "cohort_verification": cohorts, "unchanged_signed_peak_verification": unchanged,
        "protected_inputs_before": before, "protected_inputs_after": after,
        "upstream_scientific_stages_run": [], "upstream_scientific_step_calls": 0,
        "hierarchy_statistics_computed": True, "scalar_or_excel_outputs_written": 0,
        "new_cells_started": 0, "previous_summary_id": old.get("summary_id"),
        "publication_complete": False,
    }
    storage.atomic_json(control / "intra_equal_cell_v2_state.json", report)
    return report


def load_saved_node(root, record):
    """Deserialize saved arrays for rendering; never average or normalize."""
    import numpy as np
    data = root / record["data_relative"]
    metadata = read(data / "metadata.json")
    if not read(data / "COMPLETE.json").get("complete"):
        raise ValueError("Scientific source is uncommitted")
    grouped = {}
    with gzip.open(data / "curves_long.csv.gz", "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["spatial_group"], row["normalization"],
                   row["series_kind"], row["series_id"])
            grouped.setdefault(key, []).append(row)
    groups = {g: {v: {"series": {}, "n": p["n"], "mean": None,
                     "sd": None, "sem": None, "series_sd": {},
                     "series_sem": {}, "series_n": {}}
                  for v, p in views.items()}
              for g, views in metadata["series_metadata"].items()}
    x = np.asarray(metadata["distance_mid_um"], dtype=float)
    for (group, view, kind, sid), rows in grouped.items():
        rows.sort(key=lambda r: float(r["distance_mid_um"]))
        if len(rows) != 300 or not np.array_equal(
                np.asarray([float(r["distance_mid_um"]) for r in rows]), x):
            raise ValueError("Saved curve grid mismatch")
        values = np.asarray([float(r["value"]) for r in rows])
        if not np.isfinite(values).all():
            raise ValueError("Saved curve contains nonfinite values")
        arrays = {}
        for field in ("SD", "SEM"):
            entries = [r[field] for r in rows]
            if all(v == "" for v in entries):
                arrays[field] = None
            elif any(v == "" for v in entries):
                raise ValueError("Partial saved uncertainty")
            else:
                arrays[field] = np.asarray([float(v) for v in entries])
        part = groups[group][view]
        if kind == "mean":
            part.update(mean=values, sd=arrays["SD"], sem=arrays["SEM"])
        elif kind == "member":
            part["series"][sid] = values
            if arrays["SD"] is not None:
                part["series_sd"][sid] = arrays["SD"]
            if arrays["SEM"] is not None:
                part["series_sem"][sid] = arrays["SEM"]
            if rows[0]["n"] != "":
                part["series_n"][sid] = int(float(rows[0]["n"]))
        else:
            raise ValueError("Unknown saved series kind")
    return dict(key=metadata["node_key"], level=metadata["level"],
                molecule=metadata["molecule"], condition=metadata["condition"],
                BR=metadata["BR"], progress=metadata["progress"],
                blocked=metadata["blocked"], available=metadata["available"],
                groups=groups, x=x, edges=np.asarray(metadata["edges_um"]),
                directory=data.parent.parent)


def publish(report, root, control, storage):
    import intra_summary_plots as plots
    import render_on_notoverlays as overlays
    overlays.CONTROL, overlays.DATA_ROOT = control, root
    verify_protected(root, report["protected_inputs_after"])
    for record in report["nodes"]:
        data = root / record["data_relative"]
        node = load_saved_node(root, record)
        manifest = plots.render_node(node, Path(record["figures"]))
        storage.atomic_json(data / "plot_manifest.json", manifest)
        overlay = overlays.render_node(
            overlays.load_node(node["directory"], report["summary_id"]), report["summary_id"])
        storage.atomic_json(data / "PUBLICATION.json", {
            "complete": True, "published_utc": utc(), "plot_manifest": manifest,
            "overlay": overlay, "scientific_step_calls": 0,
            "hierarchy_statistics_calls": 0,
        })
        record["status"] = "updated"
        print(report["molecule"] + " | published " + record["key"], flush=True)
    storage.publish_global_index(report, report["molecule"])
    report["protected_inputs_after_publication"] = verify_protected(
        root, report["protected_inputs_after"])
    report.update(publication_complete=True, published_utc=utc(),
                  publication_scientific_step_calls=0)
    backup = control / "intra_summary_status_before_equal_cell_v2.json"
    status = control / "intra_summary_status.json"
    if status.is_file() and not backup.exists():
        storage.atomic_json(backup, read(status))
    storage.atomic_json(control / "intra_equal_cell_v2_state.json", report)
    storage.atomic_json(status, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--completed-only", required=True, action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compute-only", action="store_true")
    mode.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    config_path, config, root, control, release, storage = configure(args.config)
    with storage.SummaryLock():
        if args.publish_only:
            report = read(control / "intra_equal_cell_v2_state.json")
            if report["method"]["method_version"] != "2":
                raise ValueError("Saved hierarchy method is not v2")
            report = publish(report, root, control, storage)
        else:
            report = compute(config_path, config, root, control, release, storage)
            if not args.compute_only:
                report = publish(report, root, control, storage)
        log = control / "intra_equal_cell_v2.log"
        with log.open("a", encoding="utf-8") as stream:
            stream.write(utc() + " " + json.dumps({
                "summary_id": report["summary_id"],
                "nodes": len(report["nodes"]),
                "publication_complete": report["publication_complete"],
                "upstream_scientific_step_calls": 0,
                "scalar_or_excel_outputs_written": 0,
            }, sort_keys=True) + "\n")
        print(json.dumps({
            "summary_id": report["summary_id"], "nodes": len(report["nodes"]),
            "discovery": report["discovery"],
            "publication_complete": report["publication_complete"],
            "upstream_scientific_step_calls": 0,
            "scalar_or_excel_outputs_written": 0,
        }, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
