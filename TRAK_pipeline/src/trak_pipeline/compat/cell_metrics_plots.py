"""Plot saved On fractions and ROI mask areas without rerunning science.

Public interface: render_node(node, figures_dir, plot_limits) -> JSON metadata.
Nodes may be mappings or attribute-based objects; all numeric inputs are saved
metrics supplied by the independent aggregation core.
"""
from __future__ import annotations

import hashlib
import html
import math
import os
from pathlib import Path
import uuid

METRICS = (
    ("on_percentage", "On percentage", "%"),
    ("first_mask_ROI_area_um2", "First-mask ROI area", "µm²"),
    ("mean_ROI_area_um2", "Mean ROI area", "µm²"),
)
COLORS = {"WT": "#2166ac", "TDK": "#d95f02", "MDK": "#1b9e77"}
BR_MARKERS = {"BR1": "o", "BR2": "^", "BR3": "s"}


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _value(record, metric):
    valid = _get(record, "valid_metrics", {})
    if isinstance(valid, dict) and valid.get(metric) is False:
        return None
    return _number(_get(record, metric))


def _summary(node, metric):
    return _get(node, "metric_summaries", {}).get(metric, {})


def _jitter(identity, width=0.15):
    integer = int(hashlib.sha256(str(identity).encode("utf-8")).hexdigest()[:12], 16)
    return ((integer / float(16**12 - 1)) * 2 - 1) * width


def _cell_label(record):
    return str(_get(record, "cell", _get(record, "sample_id", "Cell")))


def _br(record):
    return str(_get(record, "BR", _get(record, "br", "Unknown BR")))


def _condition(record):
    return str(_get(record, "condition", _get(record, "genotype", "Unknown")))


def _fmt(value, decimals=2):
    value = _number(value)
    return "Unavailable" if value is None else f"{value:,.{decimals}f}"


def _stats_caption(stats, level):
    n = _get(stats, "n", 0)
    unit = "BRs" if level in ("condition", "comparison") else "cells"
    return f"n = {n} {unit}; mean = {_fmt(_get(stats, 'mean'))}"


def _mean_band(ax, stats, xmin, xmax, color="black"):
    mean = _number(_get(stats, "mean"))
    sd = _number(_get(stats, "sd"))
    n = int(_get(stats, "n", 0) or 0)
    if mean is None:
        return False
    ax.hlines(mean, xmin, xmax, color=color, linewidth=2.0, zorder=4)
    if n >= 2 and sd is not None:
        ax.fill_between([xmin, xmax], [mean - sd] * 2, [mean + sd] * 2,
                        color=color, alpha=0.08, zorder=0)
    return True


def _empty(ax, reasons="No valid saved value"):
    ax.text(0.5, 0.52, reasons, ha="center", va="center", transform=ax.transAxes,
            fontsize=12, color="#666666", wrap=True)


def _cell_panels(axes, node):
    records = _get(node, "cell_records", [])
    record = records[0] if records else {}
    metadata = {}
    for ax, (metric, title, unit) in zip(axes, METRICS):
        value = _value(record, metric)
        stats = _summary(node, metric)
        if value is None:
            value = _number(_get(stats, "mean"))
        ax.set_axis_off()
        ax.text(0.5, 0.87, title, ha="center", transform=ax.transAxes,
                fontsize=15, fontweight="bold", color="#213343")
        ax.text(0.5, 0.53, _fmt(value) + (f" {unit}" if value is not None else ""),
                ha="center", va="center", transform=ax.transAxes, fontsize=31,
                color=COLORS.get(_condition(record), "#2878b5"), fontweight="bold")
        if metric == "on_percentage":
            denominator = _get(record, "on_denominator")
            detail = f"On / (On + Not-on)\n{_get(record, 'On', 'NA'):,} / {denominator:,}" if isinstance(_get(record, "On"), (int, float)) and isinstance(denominator, (int, float)) else "On / (On + Not-on)"
        elif metric == "first_mask_ROI_area_um2":
            detail = "Block 0: original frames 0–49\nROI foreground; no error expansion"
        else:
            detail = "Time mean: original frames 0–2499\nROI foreground; no error expansion"
        ax.text(0.5, 0.22, detail, ha="center", va="center", transform=ax.transAxes,
                fontsize=11, color="#52606d", linespacing=1.5)
        metadata[metric] = {"value": value, "unit": unit}
    return metadata


def _br_panels(axes, node, limits):
    records = list(_get(node, "cell_records", []))
    metadata = {}
    for ax, (metric, title, unit) in zip(axes, METRICS):
        points = []
        for x, record in enumerate(records):
            value = _value(record, metric)
            if value is not None:
                ax.scatter(x, value, s=39, color=COLORS.get(_condition(record), "#2878b5"),
                           edgecolor="white", linewidth=0.5, zorder=3)
                points.append({"sample_id": _get(record, "sample_id"), "value": value})
        stats = _summary(node, metric)
        if not _mean_band(ax, stats, -0.4, max(0.4, len(records) - 0.6)):
            _empty(ax)
        ax.set_xticks(range(len(records)), [_cell_label(r) for r in records], rotation=55,
                      ha="right", fontsize=9)
        ax.set_xlim(-0.7, max(0.7, len(records) - 0.3))
        _axis(ax, title, unit, limits.get(metric), _stats_caption(stats, "BR"))
        metadata[metric] = {"points": points, "statistics": stats}
    return metadata


def _condition_panels(axes, node, limits):
    records = list(_get(node, "cell_records", []))
    br_records = list(_get(node, "BR_records", []))
    br_ids = sorted({_br(r) for r in records} | {_br(r) for r in br_records})
    metadata = {}
    for ax, (metric, title, unit) in zip(axes, METRICS):
        points, means, labels = [], [], []
        for x, br_id in enumerate(br_ids):
            members = [r for r in records if _br(r) == br_id]
            good = [(r, _value(r, metric)) for r in members if _value(r, metric) is not None]
            labels.append(f"{br_id}\n{len(good)} cells")
            for record, value in good:
                ax.scatter(x + _jitter(_get(record, "sample_id")), value, s=27,
                           color=COLORS.get(_condition(record), "#2878b5"), alpha=0.48,
                           edgecolor="none", zorder=2)
                points.append({"sample_id": _get(record, "sample_id"), "BR": br_id, "value": value})
            candidates = [r for r in br_records if _br(r) == br_id]
            stats = _get(candidates[0], "metrics", {}).get(metric, {}) if candidates else {}
            mean = _number(_get(stats, "mean"))
            if mean is not None:
                ax.scatter(x, mean, s=90, marker="D", color="#122a3d", edgecolor="white",
                           linewidth=0.8, zorder=5)
                means.append({"BR": br_id, "mean": mean})
        stats = _summary(node, metric)
        if not _mean_band(ax, stats, -0.4, max(0.4, len(br_ids) - 0.6)):
            _empty(ax)
        ax.set_xticks(range(len(br_ids)), labels)
        ax.set_xlim(-0.55, max(0.55, len(br_ids) - 0.45))
        _axis(ax, title, unit, limits.get(metric), _stats_caption(stats, "condition"))
        metadata[metric] = {"points": points, "BR_means": means, "statistics": stats}
    return metadata


def _comparison_panels(axes, node, limits):
    records = list(_get(node, "cell_records", []))
    br_records = list(_get(node, "BR_records", []))
    condition_records = list(_get(node, "condition_records", []))
    conditions = [c for c in ("WT", "TDK", "MDK") if any(_condition(r) == c for r in records + condition_records)]
    conditions += sorted({_condition(r) for r in records + condition_records} - set(conditions))
    metadata = {}
    for ax, (metric, title, unit) in zip(axes, METRICS):
        points, br_means, cond_means = [], [], []
        for x, condition in enumerate(conditions):
            members = [r for r in records if _condition(r) == condition]
            good = [(r, _value(r, metric)) for r in members if _value(r, metric) is not None]
            for record, value in good:
                ax.scatter(x + _jitter(_get(record, "sample_id"), 0.20), value,
                           s=19, color=COLORS.get(condition, "#777777"), alpha=0.26,
                           edgecolor="none", zorder=2)
                points.append({"sample_id": _get(record, "sample_id"), "value": value})
            for record in [r for r in br_records if _condition(r) == condition]:
                stats = _get(record, "metrics", {}).get(metric, {})
                mean = _number(_get(stats, "mean"))
                if mean is None:
                    continue
                br_id = _br(record)
                offset = {"BR1": -0.11, "BR2": 0.0, "BR3": 0.11}.get(br_id, 0.0)
                ax.scatter(x + offset, mean, s=72, marker=BR_MARKERS.get(br_id, "D"),
                           color=COLORS.get(condition, "#777777"), edgecolor="white",
                           linewidth=0.8, zorder=4)
                br_means.append({"condition": condition, "BR": br_id, "mean": mean})
            found = [r for r in condition_records if _condition(r) == condition]
            stats = _get(found[0], "metrics", {}).get(metric, {}) if found else {}
            mean, sd = _number(_get(stats, "mean")), _number(_get(stats, "sd"))
            if mean is not None:
                n = int(_get(stats, "n", 0) or 0)
                ax.errorbar(x, mean, yerr=sd if n >= 2 and sd is not None else None,
                            fmt="_", color="black", markersize=25, markeredgewidth=2.6,
                            linewidth=1.8, capsize=5, zorder=6)
                cond_means.append({"condition": condition, "statistics": stats})
        if not cond_means:
            _empty(ax)
        labels = [f"{c}\n{sum(1 for r in records if _condition(r) == c and _value(r, metric) is not None)} cells" for c in conditions]
        ax.set_xticks(range(len(conditions)), labels)
        ax.set_xlim(-0.55, max(0.55, len(conditions) - 0.45))
        _axis(ax, title, unit, limits.get(metric), "Black: equal-BR mean ± 1 BR sample SD")
        metadata[metric] = {"points": points, "BR_means": br_means, "condition_statistics": cond_means}
    return metadata


def _axis(ax, title, unit, limit, subtitle):
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_ylabel(unit, fontsize=12)
    if limit is not None:
        ax.set_ylim(float(limit[0]), float(limit[1]))
    ax.grid(axis="y", color="#e6e9ed", linewidth=0.8, zorder=-1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9, color="#52606d")


def _atomic_text(path, value):
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_node(node, figures_dir, plot_limits):
    """Write one PNG and English index atomically; return auditable plot inputs."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    level = str(_get(node, "level", "cell"))
    key = str(_get(node, "key", "Cell metrics"))
    progress = _get(node, "progress", {})
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.9), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.26, top=0.77, wspace=0.34)
    if level == "cell":
        details = _cell_panels(axes, node)
    elif level.lower() in ("br", "replica", "replicate"):
        details = _br_panels(axes, node, plot_limits)
    elif level == "condition":
        details = _condition_panels(axes, node, plot_limits)
    else:
        details = _comparison_panels(axes, node, plot_limits)
    molecule=str(_get(node,"molecule","TRAK1"))
    title_key=key if molecule in key else molecule+" | "+key
    fig.suptitle(f"{title_key} | On fraction and ROI mitochondrial mask area",
                 fontsize=19, fontweight="bold", x=0.5, y=0.96)
    completed, expected = _get(progress, "completed"), _get(progress, "expected")
    status = str(_get(progress, "status", ""))
    if completed is not None and expected is not None:
        line = f"{status} | completed {completed}/{expected} cells"
        if _get(progress, "failed", 0):
            line += f"; failed {_get(progress, 'failed')}"
        fig.text(0.5, 0.895, line, ha="center", fontsize=11, color="#52606d")
    if level == "cell":
        records = _get(node, "cell_records", [])
        record = records[0] if records else {}
        count_text = (f"On: {_get(record, 'On', 'NA')}   Not-on: {_get(record, 'Not_on', 'NA')}   "
                      f"Unknown: {_get(record, 'unknown', 'NA')}   QC excluded: {_get(record, 'QC_excluded', 'NA')}")
        fig.text(0.5, 0.20, count_text, ha="center", fontsize=11)
    elif level == "condition":
        handles = [Line2D([], [], marker="o", linestyle="", color="#6b7e8e", alpha=0.55, label="Individual cells"),
                   Line2D([], [], marker="D", linestyle="", color="#122a3d", label="BR mean"),
                   Line2D([], [], color="black", linewidth=2, label="Equal-BR mean; band = ±1 BR SD")]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.125), ncol=3, frameon=False, fontsize=10)
    elif level not in ("BR", "br", "replica", "replicate"):
        handles = [Line2D([], [], marker=marker, linestyle="", color="#52606d", label=br) for br, marker in BR_MARKERS.items()]
        handles.append(Line2D([], [], color="black", linewidth=2, label="Equal-BR condition mean ± 1 BR SD"))
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.125), ncol=4, frameon=False, fontsize=10)
    else:
        fig.text(0.5, 0.17, "Points: individual cells. Black line: equal-cell mean. Band: ±1 sample SD across cells.", ha="center", fontsize=10)
    footer = ("On % = 100 × On / (On + Not-on); unknown and QC excluded are outside the denominator.\n"
              "First mask covers frames 0–49 (1 s); mean area covers frames 0–2499. ROI foreground area; no error expansion.\n"
              "Spatial compatibility under unverified C1/C2 identity registration. Descriptive summaries; no significance test.")
    fig.text(0.5, 0.035, footer, ha="center", va="bottom", fontsize=9, color="#52606d", linespacing=1.45)
    destination = figures_dir / "Cell_metrics.png"
    temporary = figures_dir / (".Cell_metrics." + uuid.uuid4().hex + ".tmp.png")
    try:
        fig.savefig(temporary, dpi=170, facecolor="white", metadata={"Software": str(_get(node,"molecule","TRAK1"))+" cell metrics"})
        os.replace(temporary, destination)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()
    data_directory = _get(node, "data_directory")
    data_link = ""
    if data_directory:
        relative = os.path.relpath(Path(data_directory), figures_dir).replace(os.sep, "/")
        data_link = f'<p><a href="{html.escape(relative, quote=True)}/">Numerical results and provenance</a></p>'
    document = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{html.escape(key)} — Cell metrics</title>'
                '<style>body{font:16px system-ui,sans-serif;margin:32px auto;max-width:1500px;padding:0 24px;color:#213343}img{width:100%;height:auto}a{color:#2878b5}</style>'
                f'<h1>{html.escape(key)}: On fraction and ROI mask area</h1>'
                '<p>On percentage excludes unknown and QC-excluded localizations. First-mask area uses the 0–49 frame block. '
                'Area is measured inside the original cell ROI without localization-error expansion.</p>'
                f'<p>{html.escape(status)}</p><a href="Cell_metrics.png"><img src="Cell_metrics.png" alt="On percentage, first-mask ROI area, and mean ROI area"></a>'
                f'{data_link}<p>BR summaries give equal weight to cells; condition summaries give equal weight to BR means. '
                'SD and SEM describe the stated aggregation units. No formal significance test is performed.</p></html>')
    _atomic_text(figures_dir / "INDEX.html", document)
    return {"schema_version": 1, "level": level, "key": key, "png": "Cell_metrics.png",
            "index": "INDEX.html", "metrics": details, "plot_limits": plot_limits,
            "mean_weighting": "equal cells within BR; equal BRs within condition",
            "uncertainty": "sample SD (ddof=1), shown only for at least two aggregation units",
            "on_denominator": "On + Not-on; unknown and QC excluded are not included"}
