"""PNG-only visual summaries of saved Intra curves; no scientific recomputation.

The node protocol is supplied by intra_summary_core. Normalization, cohort
selection and equal weighting belong to that module, never to this renderer.
"""
from __future__ import annotations

import html
import os
from pathlib import Path
import uuid

import numpy as np

GROUPS = ("On", "Not-on")
VIEWS = ("signed", "area", "peak")
RANGES = {"full": (0.0, 3.0), "zoom": (0.0, 0.5)}
UNITS = {"signed": "um^-1", "area": "um^-1", "peak": "dimensionless"}
VIEW_LABELS = {
    "signed": "Signed Intra",
    "area": "Signed-area-normalized Intra",
    "peak": "Positive-peak-normalized Intra",
}
Y_LABELS = {
    "signed": "Signed Intra density (1/um)",
    "area": "Signed-area-normalized density (1/um)",
    "peak": "Relative signed Intra (dimensionless)",
}
CONDITION_COLORS = {"WT": "#2166ac", "TDK": "#d95f02", "MDK": "#1b9e77"}


def _curve(value, n):
    if value is None:
        return None
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (n,) or not np.all(np.isfinite(out)):
        raise ValueError("Plot curves must be finite vectors matching the fixed bins")
    return out


def _sd_curve(value, n):
    out = _curve(value, n)
    if out is not None and np.any(out < 0):
        raise ValueError("Sample SD cannot be negative")
    return out


def _series(part, n):
    curves = {}
    for name, value in part.get("series", {}).items():
        curve = _curve(value, n)
        if curve is not None:
            curves[str(name)] = curve
    return curves


def _ylim(curves):
    """One finite, zero-inclusive linear scale; never changes curve values."""
    chunks = [np.asarray(c, dtype=float) for c in curves if c is not None]
    if not chunks:
        return (-1.0, 1.0)
    low = min(0.0, *(float(np.min(c)) for c in chunks))
    high = max(0.0, *(float(np.max(c)) for c in chunks))
    span = high - low
    if span <= 0:
        return (-1.0, 1.0)
    pad = 0.065 * span
    return (low - pad, high + pad)


def view_limits(node, view):
    """Return the same Y range for both groups and full/zoom of one view.

    Comparison limits use every condition mean +/- available sample SD. Inner
    inspection figures also include member lines. No SD/SEM is fabricated.
    """
    n = len(node["x"])
    traces = []
    for group in GROUPS:
        part = node.get("groups", {}).get(group, {}).get(view, {})
        series = _series(part, n)
        traces.extend(series.values())
        if node["level"] == "comparison":
            sds = part.get("series_sd", {})
            for name, curve in series.items():
                sd = _sd_curve(sds.get(name), n)
                if sd is not None:
                    traces.extend((curve - sd, curve + sd))
        else:
            mean = _curve(part.get("mean"), n)
            sd = _sd_curve(part.get("sd"), n)
            if mean is not None:
                traces.append(mean)
                if node["level"] == "condition" and sd is not None:
                    traces.extend((mean - sd, mean + sd))
    return _ylim(traces)


def _progress(node):
    progress = node.get("progress", {})
    done, expected = progress.get("completed", 0), progress.get("expected", 0)
    included = progress.get("included", 0)
    status = str(progress.get("status", "PROGRESS"))
    kind = "cells"
    text = f"{status} | completed {done}/{expected} {kind} | included {included}"
    if node["level"] in ("condition", "comparison"):
        valid_brs = progress.get("valid_BRs")
        expected_brs = progress.get("expected_BRs")
        if valid_brs is not None:
            text += f" | valid BRs {valid_brs}"
            if expected_brs is not None:
                text += f"/{expected_brs}"
    return text


def _caption(node, view):
    if node["level"] == "BR":
        if view == "area":
            return ("Bold: equal-cell mean of individually signed-area-normalized curves. "
                    "Thin: each cell normalized over 0-3 um. No final re-normalization.")
        if view == "peak":
            return "Thin: each cell divided by its full-range positive peak. Bold: their equal-cell mean."
        return "Thin: individual cells. Bold: equal-cell mean signed Intra."
    if node["level"] == "condition":
        return ("Thin: included BR curves. Bold: equal-BR mean. Shading: +/-1 sample SD across BRs, "
                "shown only with at least two valid BRs.")
    return ("Each line: equal-BR condition mean. Shading: +/-1 sample SD across BRs, "
            "shown only with at least two valid BRs.")


def _footer():
    return ("Descriptive summaries; no p values or biological pairing. Signed curves may be negative.\n"
            "Nominal camera calibration; C1/C2 identity registration remains independently unverified.")


def _atomic_figure(fig, target):
    target = Path(target)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        fig.savefig(temp, format="png", dpi=180, facecolor="white")
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def _atomic_text(text, target):
    target = Path(target)
    temp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp, target)
    finally:
        if temp.exists():
            temp.unlink()


def _plot_axis(ax, node, group, view, x):
    part = node.get("groups", {}).get(group, {}).get(view, {})
    series = _series(part, len(x))
    if node["level"] == "comparison":
        sds = part.get("series_sd", {})
        shades = False
        for index, (name, curve) in enumerate(series.items()):
            color = CONDITION_COLORS.get(name, f"C{index}")
            ax.plot(x, curve, color=color, linewidth=1.9, label=name, zorder=3)
            sd = _sd_curve(sds.get(name), len(x))
            if sd is not None:
                ax.fill_between(x, curve - sd, curve + sd, color=color, alpha=0.15,
                                linewidth=0, label=f"{name} +/-1 BR sample SD", zorder=2)
                shades = True
        if not series:
            ax.text(0.5, 0.5, "No available condition curve\nSee exclusions and progress records",
                    ha="center", va="center", transform=ax.transAxes)
    else:
        for index, (name, curve) in enumerate(series.items()):
            ax.plot(x, curve, linewidth=0.8, alpha=0.45, color=f"C{index % 10}",
                    label=name, zorder=2)
        mean = _curve(part.get("mean"), len(x))
        sd = _sd_curve(part.get("sd"), len(x))
        shades = node["level"] == "condition" and mean is not None and sd is not None
        if mean is not None:
            color = "#182e49"
            if shades:
                ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15,
                                linewidth=0, label="+/-1 BR sample SD", zorder=2)
            name = "Equal-cell mean" if node["level"] == "BR" else "Equal-BR mean"
            if node["level"] == "BR" and view == "area":
                name = "Mean of cell-area-normalized curves"
            ax.plot(x, mean, color=color, linewidth=2.3, label=name, zorder=4)
        else:
            ax.text(0.5, 0.5, "No available summary curve\nSee exclusions and progress records",
                    ha="center", va="center", transform=ax.transAxes)
        if node["level"] == "condition" and mean is not None and sd is None:
            ax.text(0.98, 0.82, "BR sample SD unavailable (n < 2)",
                    ha="right", va="top", fontsize=8, transform=ax.transAxes)
    ax.axhline(0.0, color="0.5", linewidth=0.8, zorder=1)
    ax.grid(True, color="0.9", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_title(group, fontsize=12)
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel(Y_LABELS[view])
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Each member label remains readable for current ~10-cell BRs.
        columns = 2 if len(handles) > 8 else 1
        ax.legend(handles, labels, fontsize=7.0, ncol=columns, framealpha=0.9,
                  loc="upper right")
    return list(series), bool(shades)


def render_node(node, figures_dir):
    """Render six flat PNGs plus INDEX.html; return JSON-serializable metadata.

    Caller supplies a source-verified summary node and controls cache signatures.
    Atomic file replacement supports resume without partial PNG files.
    """
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if node.get("level") not in ("BR", "condition", "comparison"):
        raise ValueError("Unknown summary level")
    x = np.asarray(node["x"], dtype=float)
    if x.shape != (300,) or not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
        raise ValueError("Expected 300 increasing fixed-bin centers")
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "level": node["level"], "key": node["key"],
                "status": node.get("progress", {}).get("status", "PROGRESS"),
                "progress": node.get("progress", {}), "views": {}, "plots": []}
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 12, "axes.labelsize": 10}):
        for view in VIEWS:
            limits = view_limits(node, view)
            manifest["views"][view] = {"unit": UNITS[view], "ylim": list(limits)}
            for range_name, xlimits in RANGES.items():
                fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9), sharex=True, sharey=True)
                try:
                    ids, sd_shown = {}, {}
                    for ax, group in zip(axes, GROUPS):
                        ids[group], sd_shown[group] = _plot_axis(ax, node, group, view, x)
                        ax.set_xlim(*xlimits)
                        ax.set_ylim(*limits)
                        ax.tick_params(labelleft=True)
                    fig.suptitle(f"{node.get('molecule','')} | {node['key']} | {VIEW_LABELS[view]}", y=0.97, fontsize=14)
                    fig.text(0.5, 0.917, _progress(node), ha="center", va="center", fontsize=10,
                             color="#a34a00" if "PROGRESS" in _progress(node) else "#333333")
                    fig.text(0.5, 0.088, _caption(node, view), ha="center", va="center", fontsize=8)
                    fig.text(0.5, 0.038, _footer(), ha="center", va="center", fontsize=7.4,
                             color="0.33")
                    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.85, wspace=0.15)
                    filename = f"Intra_{view}_{range_name}.png"
                    _atomic_figure(fig, figures / filename)
                    manifest["plots"].append({"file": filename, "view": view, "range": range_name,
                                              "xlim": list(xlimits), "ylim": list(limits),
                                              "unit": UNITS[view], "series_ids": ids,
                                              "sample_sd_shown": sd_shown})
                finally:
                    plt.close(fig)
    _write_index(node, figures, manifest)
    return manifest


def _write_index(node, figures, manifest):
    title = html.escape(f"{node.get('molecule','')} | {node['key']} | Intra summaries")
    sections = []
    for view in VIEWS:
        images = []
        for plot in manifest["plots"]:
            if plot["view"] == view:
                file = html.escape(plot["file"], quote=True)
                caption = html.escape(f"{view}; {plot['xlim'][0]:g}-{plot['xlim'][1]:g} um")
                images.append(f'<figure><a href="{file}"><img src="{file}" alt="{caption}"></a>'
                              f'<figcaption>{caption}</figcaption></figure>')
        sections.append(f'<section><h2>{html.escape(VIEW_LABELS[view])}</h2>'
                        f'<p>{html.escape(_caption(node, view))}</p><div class="pair">'
                        + "".join(images) + "</div></section>")
    exclusion_count = len(node.get("exclusions", []))
    content = ("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\">"
               f"<title>{title}</title><style>body{{font-family:Arial,sans-serif;margin:24px;"
               "max-width:1800px;color:#222}}.pair{display:flex;gap:14px;flex-wrap:wrap}"
               "figure{margin:0;flex:1;min-width:420px}img{width:100%;height:auto}"
               "figcaption{font-size:14px}.status{font-weight:bold;color:#a34a00}"
               "section{margin-top:28px}footer{margin-top:24px;color:#555}</style>"
               f"<h1>{title}</h1><p class=\"status\">{html.escape(_progress(node))}</p>"
               f"<p>Membership exclusions recorded: {exclusion_count}. "
               "Source and membership tables are retained in the matching analysis_results directory.</p><p><a href='On_NotOn_Overlay/INDEX.html'>On / Not-on on the same axes: full and zoom views</a></p>"
               + "".join(sections) + f"<footer>{html.escape(_footer()).replace(chr(10), '<br>')}"
               "<br>All normalizations use the complete 0-3 um range before plotting the zoom. "
               "Negative bins are retained; these are signed shapes, not nonnegative PDFs.</footer></html>")
    _atomic_text(content, figures / "INDEX.html")
