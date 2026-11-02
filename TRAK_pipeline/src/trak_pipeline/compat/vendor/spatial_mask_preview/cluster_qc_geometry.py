"""Exact pixel-square distances and frame-presence diagnostics for image QC.

All distances use native image pixels. Pixel (x, y) occupies the closed square
[x - 0.5, x + 0.5] x [y - 0.5, y + 0.5]. This module does not alter masks,
classify localization errors, associate observations, or link trajectories.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree


class MaskSquareDistance:
    """Reusable distance index for the union of foreground pixel squares.

    A nearest-center query alone is insufficient: a different pixel can have
    the nearest square. For nearest-center distance d, a square improving the
    answer must have its center within d + sqrt(0.5). The added term is the
    half diagonal of one pixel. Every center within that guaranteed radius is
    evaluated using the analytic closed-square distance, without rasterizing
    continuous query positions.
    """

    def __init__(self, mask: np.ndarray):
        foreground = np.asarray(mask, dtype=bool)
        if foreground.ndim != 2:
            raise ValueError("mask must have two dimensions (Y, X)")
        yy, xx = np.nonzero(foreground)
        self.centers_xy = np.column_stack((xx, yy)).astype(np.float64)
        self.tree = cKDTree(self.centers_xy) if len(xx) else None

    def distances(self, points_xy: np.ndarray) -> np.ndarray:
        """Return exact distances for a finite array of shape (N, 2).

        Foreground interiors, edges and corners have distance zero. An empty
        foreground produces positive infinity for every query. Invalid query
        coordinates raise ValueError, allowing callers to handle unknown
        positions explicitly rather than treating them as background.
        """
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points_xy must have shape (N, 2)")
        if not np.isfinite(points).all():
            raise ValueError("query coordinates must be finite")
        result = np.full(len(points), np.inf, dtype=np.float64)
        if not len(points) or self.tree is None:
            return result

        center_distances, nearest = self.tree.query(points, k=1)
        # Round the search radius outward so boundary candidates are included.
        radii = np.nextafter(center_distances + math.sqrt(0.5), np.inf)
        candidates = self.tree.query_ball_point(points, radii)
        for index, (point, neighborhood) in enumerate(zip(points, candidates)):
            # Retain the nearest center even if a library's radius comparison
            # incurs a floating-point discrepancy at extreme magnitudes.
            centers = self.centers_xy[
                np.append(neighborhood, int(nearest[index])).astype(np.intp)
            ]
            excess = np.maximum(np.abs(centers - point) - 0.5, 0.0)
            result[index] = np.sqrt(np.min(np.sum(excess * excess, axis=1)))
        return result

    def distance(self, x: float, y: float) -> float:
        """Scalar form of :meth:`distances`, in native pixels."""
        return float(self.distances(np.array([[x, y]], dtype=np.float64))[0])


def nearest_mask_distances(mask: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Convenience wrapper; reuse MaskSquareDistance for repeated queries."""
    return MaskSquareDistance(mask).distances(points_xy)


def longest_occupied_frame_run(frames: np.ndarray) -> dict[str, int | None]:
    """Summarize the longest consecutive run of occupied original frames.

    Duplicate point rows count only once. Gaps remain gaps. Ties select the
    earliest run; an empty input returns None, None, 0. This describes frame
    presence only and does not assert observation or molecule identity.
    """
    original = np.asarray(frames)
    if original.ndim != 1:
        raise ValueError("frames must be a one-dimensional sequence")
    if not len(original):
        return {"start_frame": None, "end_frame": None, "length": 0}
    if original.dtype.kind not in "iuf":
        raise ValueError("frame numbers must be finite nonnegative integers")
    if (
        not np.isfinite(original).all()
        or np.any(original < 0)
        or np.any(original != np.floor(original))
        or np.any(original >= 2**63)
    ):
        raise ValueError("frame numbers must be finite nonnegative integers")
    occupied = np.unique(original.astype(np.int64))
    starts = np.r_[0, np.flatnonzero(np.diff(occupied) != 1) + 1]
    ends = np.r_[starts[1:] - 1, len(occupied) - 1]
    lengths = ends - starts + 1
    winner = int(np.argmax(lengths))
    return {
        "start_frame": int(occupied[starts[winner]]),
        "end_frame": int(occupied[ends[winner]]),
        "length": int(lengths[winner]),
    }
