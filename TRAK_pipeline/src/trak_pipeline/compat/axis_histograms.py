"""Render separate, per-axis nominal-SE histograms from saved 1D counts.

The X and Y ranges are applied independently. This never marginalizes the
jointly cropped 2D histogram, and never reads raw images or refits positions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

GROUPS = ("All", "On", "Not-on")
FIELDS = (("x_se_nm", "X"), ("y_se_nm", "Y"))
COLORS = {"All": "#555555", "On": "#00B894", "Not-on": "#A45BDB"}
FILENAMES = {"X": "SE_X_All_On_NotOn_linear.png", "Y": "SE_Y_All_On_NotOn_linear.png"}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load_counts(counts_npz, metadata_json):
    metadata = json.loads(Path(metadata_json).read_text(encoding="utf-8"))
    expected = np.arange(0, 205, 5, dtype=float)
    counts = {}
    summaries = {}
    with np.load(counts_npz, allow_pickle=False) as archive:
        edges = np.asarray(archive["SE_edges_nm"], dtype=float)
        if not np.array_equal(edges, expected):
            raise ValueError("Expected fixed 0-200 nm edges at 5 nm intervals")
        for field, _axis in FIELDS:
            for group in GROUPS:
                key = field + "__" + group
                values = np.asarray(archive[key])
                if values.shape != (40,) or not np.issubdtype(values.dtype, np.integer) or np.any(values < 0):
                    raise ValueError("Invalid saved nonnegative 40-bin counts: " + key)
                counts[key] = values.astype(np.int64, copy=True)
                info = metadata["outside_counts"][key]
                summary = {name: int(info[name]) for name in ("total", "displayed", "below_range", "above_range", "nonfinite")}
                if any(n < 0 for n in summary.values()):
                    raise ValueError("Negative histogram summary count: " + key)
                if int(values.sum()) != summary["displayed"]:
                    raise ValueError("Saved bin-count sum differs from displayed count: " + key)
                if summary["total"] != summary["displayed"] + summary["below_range"] + summary["above_range"] + summary["nonfinite"]:
                    raise ValueError("Histogram summary does not conserve rows: " + key)
                summaries[key] = summary
    return edges.copy(), counts, summaries


def render_axis_histograms(counts_npz, metadata_json, figures_dir, sample_name):
    """Save X and Y three-panel PNGs using exact previously saved 1D counts.

    All, On and Not-on are shown as separate panels. All six panels share one
    linear count limit. Values outside the plotted axis range are only omitted
    from that axis's display, not from selection or the other axis histogram.
    """
    counts_npz = Path(counts_npz)
    metadata_json = Path(metadata_json)
    figures_dir = Path(figures_dir)
    if not counts_npz.is_file() or not metadata_json.is_file():
        raise FileNotFoundError("Saved histogram counts and metadata are required")
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {"counts_npz_sha256": _sha(counts_npz), "metadata_json_sha256": _sha(metadata_json)}
    edges, counts, summaries = _load_counts(counts_npz, metadata_json)
    maximum = max(1, max(int(values.max()) for values in counts.values()))
    ylim = (0.0, float(maximum) * 1.12)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pictures = []
    for field, axis_name in FIELDS:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), sharex=True, sharey=True)
        try:
            for axis, group in zip(axes, GROUPS):
                key = field + "__" + group
                info = summaries[key]
                outside = info["below_range"] + info["above_range"]
                axis.stairs(counts[key], edges, color=COLORS[group], linewidth=1.8, fill=False)
                axis.set(xlabel=axis_name + " nominal model SE (nm)", ylabel="Localizations per 5 nm bin",
                         xlim=(0, 200), ylim=ylim,
                         title=f"{group} | shown {info['displayed']:,}; outside {outside:,}; NA {info['nonfinite']:,}")
                axis.set_xscale("linear")
                axis.set_yscale("linear")
                axis.grid(alpha=0.2)
            fig.suptitle(str(sample_name) + " | " + axis_name + " nominal model SE | no precision threshold")
            fig.text(0.5, 0.025,
                     "Manufacturer-nominal calibration; model SE is not validated accuracy. "
                     "Axis ranges applied independently; repeated localizations retained.",
                     ha="center", fontsize=9)
            fig.tight_layout(rect=(0, 0.065, 1, 0.92))
            target = figures_dir / FILENAMES[axis_name]
            temporary = target.with_name("." + target.name + ".tmp.png")
            try:
                fig.savefig(temporary, dpi=180, format="png")
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
            pictures.append({"axis": axis_name, "name": target.name, "sha256": _sha(target), "bytes": target.stat().st_size})
        finally:
            plt.close(fig)
    if source_hashes != {"counts_npz_sha256": _sha(counts_npz), "metadata_json_sha256": _sha(metadata_json)}:
        raise ValueError("Source histogram files changed while rendering")
    manifest = {
        "schema_version": 1, "sample_name": str(sample_name),
        "source_counts_npz": str(counts_npz), "source_metadata_json": str(metadata_json),
        **source_hashes, "SE_edges_nm": edges.tolist(), "groups": list(GROUPS),
        "shared_linear_y_limits": list(ylim), "maximum_saved_bin_count": maximum,
        "outside_counts": summaries, "images": pictures,
        "per_axis_independent_range": True,
        "joint_2D_crop_not_used": True,
        "selection_changed": False, "no_fitting_or_TARDIS": True,
        "calibration_status": "manufacturer_nominal", "validated_accuracy": False,
        "main_index_modified": False,
    }
    destination = figures_dir / "axis_histograms_manifest.json"
    temporary = destination.with_name("." + destination.name + ".tmp")
    try:
        temporary.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--figures", required=True, type=Path)
    parser.add_argument("--sample-name", required=True)
    args = parser.parse_args()
    manifest = render_axis_histograms(args.counts, args.metadata, args.figures, args.sample_name)
    print(json.dumps({"figures": [row["name"] for row in manifest["images"]],
                      "shared_linear_y_limits": manifest["shared_linear_y_limits"]}))


if __name__ == "__main__":
    main()
