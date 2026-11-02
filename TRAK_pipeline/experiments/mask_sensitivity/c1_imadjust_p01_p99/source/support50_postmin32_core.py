"""Support50 recipe with min32 applied only after temporal support selection.

No per-frame foreground-size cleanup is performed here. Raw Nellie broad
foreground is counted at each pixel across 50 original frames, thresholded
at >=10, cleaned across the full image with 8-connected components >=32,
then filled at enclosed 4-connected background components strictly <64.
ROI clipping, registration, localization and file I/O are outside this module.
"""

from __future__ import annotations

import operator

import numpy as np
from scipy import ndimage as ndi

FG8 = ndi.generate_binary_structure(2, 2)
BG4 = ndi.generate_binary_structure(2, 1)


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(name + " must be a positive integer") from error
    if result < 1:
        raise ValueError(name + " must be a positive integer")
    return result


def _binary_image(mask):
    result = np.asarray(mask, dtype=bool)
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError("mask must be a nonempty two-dimensional full image")
    return result


def clean_frame(mask, minimum=32):
    """Keep whole full-image 8-connected components with area >=minimum.

    Return (clean_mask, uint32_labels, component_areas, keep_flags), matching
    the recorded union_core.clean_frame convention. Label zero is background
    and never retained. Components touching image edges receive no exemption.
    """
    foreground = _binary_image(mask)
    minimum = _positive_integer(minimum, "minimum")
    labels, count = ndi.label(foreground, structure=FG8)
    area = np.bincount(labels.ravel(), minlength=count + 1)
    keep = area >= minimum
    keep[0] = False
    return keep[labels], labels.astype(np.uint32), area, keep


def fill_enclosed(mask, exclusive_limit=64):
    """Fill enclosed full-image 4-connected background areas <limit.

    Return (final_mask, added_mask, uint32_background_labels, areas,
    edge_flags, enclosed_flags, fill_flags), matching recorded union_core.
    A component of exactly64 is preserved when exclusive_limit=64. All
    edge-connected background remains unfilled; existing foreground is kept.
    """
    foreground = _binary_image(mask)
    exclusive_limit = _positive_integer(exclusive_limit, "exclusive_limit")
    labels, count = ndi.label(~foreground, structure=BG4)
    area = np.bincount(labels.ravel(), minlength=count + 1)
    edge = np.zeros(count + 1, dtype=bool)
    edge[np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])] = True
    edge[0] = False
    enclosed = ~edge
    enclosed[0] = False
    fill = enclosed & (area < exclusive_limit)
    added = fill[labels]
    return foreground | added, added, labels.astype(np.uint32), area, edge, enclosed, fill


def process_counts(
    raw_foreground_count, support_minimum=10, minimum_foreground=32,
    hole_limit=64, frame_count=50,
):
    """Process a raw-broad per-pixel temporal count map without per-frame min32.

    Input is a nonnegative integer YX map with values <=frame_count. Arrays
    are returned in a dictionary; the input is not modified or shared with
    the returned foreground_count. No ROI clipping or extra voting occurs.
    """
    frame_count = _positive_integer(frame_count, "frame_count")
    support_minimum = _positive_integer(support_minimum, "support_minimum")
    minimum_foreground = _positive_integer(minimum_foreground, "minimum_foreground")
    hole_limit = _positive_integer(hole_limit, "hole_limit")
    if frame_count > 255 or support_minimum > frame_count:
        raise ValueError("require support_minimum <= frame_count <= 255")
    counts = np.asarray(raw_foreground_count)
    if counts.ndim != 2 or min(counts.shape) < 1:
        raise ValueError("raw_foreground_count must be a nonempty full-image YX array")
    if counts.dtype.kind not in "bui" or np.any(counts < 0) or np.any(counts > frame_count):
        raise ValueError("raw counts must be integers between zero and frame_count")
    counts = counts.astype(np.uint8, copy=True)
    supported = counts >= support_minimum
    cleaned, labels, areas, kept = clean_frame(supported, minimum_foreground)
    final, added, bg_labels, bg_areas, edge, enclosed, filled = fill_enclosed(cleaned, hole_limit)
    return {
        "foreground_count": counts,
        "support_mask": supported,
        "post_min32_mask": cleaned,
        "final_mask": final,
        "filled_holes": added,
        "support_component_labels": labels,
        "support_component_areas": areas,
        "support_component_keep": kept,
        "background_component_labels": bg_labels,
        "background_component_areas": bg_areas,
        "background_component_edge": edge,
        "background_component_enclosed": enclosed,
        "background_component_fill": filled,
    }


def process_block(raw_masks, support_minimum=10, minimum_foreground=32, hole_limit=64):
    """Count exactly50 ordered raw-broad YX masks, then apply process_counts.

    Pages are assumed to be the caller's verified consecutive original time
    frames. This pure function does not infer or reorder temporal mappings.
    """
    foreground = np.asarray(raw_masks, dtype=bool)
    if foreground.ndim != 3 or foreground.shape[0] != 50 or min(foreground.shape[1:]) < 1:
        raise ValueError("raw_masks must contain exactly50 nonempty full-image YX masks")
    counts = np.count_nonzero(foreground, axis=0).astype(np.uint8)
    return process_counts(counts, support_minimum, minimum_foreground, hole_limit, frame_count=50)
