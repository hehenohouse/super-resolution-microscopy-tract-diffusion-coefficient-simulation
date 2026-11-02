"""Fixed-pipeline mask-stage diagnostics, without biological attribution.

Pixel geometry matches geometry.py: integer pixel centers represent closed
unit squares; an axis-aligned localization ellipse includes tangency. The
stage reason is the first failure in the specified processing sequence and
counterfactual raw-support threshold. It is not a causal biological label.
"""

from __future__ import annotations

import math

import numpy as np


def ellipse_support_max(count_map, x, y, a, b, tol=1e-12):
    """Maximum support count over pixel squares intersecting an ellipse.

    The ellipse has center (x, y), positive semiaxes (a, b), and native-pixel
    units. A square at integer (px, py) intersects when
    (max(abs(x-px)-0.5,0)/a)**2 + (max(abs(y-py)-0.5,0)/b)**2 <= 1+tol.
    The search window expands by sqrt(1+tol), retaining tolerance contacts.

    count_map must be a two-dimensional nonnegative integer array (boolean
    is accepted). No intersecting in-image square, or all-zero support,
    returns zero. Portions outside the image do not contribute support;
    callers retain responsibility for separate unknown-region checks.
    """
    counts = np.asarray(count_map)
    if counts.ndim != 2:
        raise ValueError("count_map must be a two-dimensional YX array")
    if counts.dtype.kind not in "bui":
        raise ValueError("support counts must be nonnegative integers")
    if counts.dtype.kind == "i" and np.any(counts < 0):
        raise ValueError("support counts must be nonnegative integers")
    if not np.isfinite([x, y, a, b]).all() or a <= 0 or b <= 0:
        raise ValueError("ellipse coordinates must be finite and axes positive")
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("tol must be finite and nonnegative")

    height, width = counts.shape
    rx, ry = a * math.sqrt(1 + tol), b * math.sqrt(1 + tol)
    x0 = max(0, math.ceil(x - rx - 0.5 - 1e-12))
    x1 = min(width - 1, math.floor(x + rx + 0.5 + 1e-12))
    y0 = max(0, math.ceil(y - ry - 0.5 - 1e-12))
    y1 = min(height - 1, math.floor(y + ry + 0.5 + 1e-12))
    if x1 < x0 or y1 < y0:
        return 0

    local = counts[y0:y1 + 1, x0:x1 + 1]
    # Zero-valued squares cannot increase the maximum, so their geometry need
    # not be evaluated. This is an optimization, not a foreground threshold.
    yy, xx = np.nonzero(local)
    if not len(xx):
        return 0
    dx = np.maximum(np.abs((xx + x0) - x) - 0.5, 0.0)
    dy = np.maximum(np.abs((yy + y0) - y) - 0.5, 0.0)
    contact = (dx / a) ** 2 + (dy / b) ** 2 <= 1 + tol
    return int(local[yy[contact], xx[contact]].max()) if contact.any() else 0


def first_failure_reason(
    raw_ge10_hit, clean_ge10_hit, post_min32_hit, final_hit, raw_any_hit
):
    """Describe final contact or the first fixed-pipeline mask-stage failure.

    Inputs are boolean ellipse-intersection flags. Required set nesting is
    raw_ge10 within raw_any, clean_ge10 within raw_ge10, post_min32 within
    clean_ge10, and post_min32 within final. Final may include newly filled
    holes even when every earlier flag is false. Impossible flag nesting
    raises ValueError before any reason is returned.

    Names refer to the fixed raw>=10 counterfactual and actual pipeline
    order; they do not identify molecular histories or biological causes.
    """
    flags = [raw_ge10_hit, clean_ge10_hit, post_min32_hit, final_hit, raw_any_hit]
    if not all(isinstance(value, (bool, np.bool_)) for value in flags):
        raise ValueError("stage hit flags must be boolean")
    implications = [
        (raw_ge10_hit, raw_any_hit, "raw_ge10 must be within raw_any"),
        (clean_ge10_hit, raw_ge10_hit, "clean_ge10 must be within raw_ge10"),
        (post_min32_hit, clean_ge10_hit, "post_min32 must be within clean_ge10"),
        (post_min32_hit, final_hit, "post_min32 must be within final"),
    ]
    for earlier, containing, explanation in implications:
        if earlier and not containing:
            raise ValueError("impossible stage nesting: " + explanation)

    if final_hit:
        return "final_compatible"
    if clean_ge10_hit:
        return "post_support_min32_removed"
    if raw_ge10_hit:
        return "per_frame_min32_lowered_below_threshold"
    if raw_any_hit:
        return "raw_broad_present_below_per_pixel_support_threshold"
    return "no_raw_broad_foreground_in_ellipse_in_block"
