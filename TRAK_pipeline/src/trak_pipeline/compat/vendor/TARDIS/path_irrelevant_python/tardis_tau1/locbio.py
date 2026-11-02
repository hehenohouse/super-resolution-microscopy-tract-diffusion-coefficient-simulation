"""Experimental standalone reader for the observed nim_locb v5/74-byte layout.

This is an empirically inferred layout, not an ONI format specification.
Only frame/channel and the two candidate coordinate fields are interpreted.
All other bytes are preserved. Stored coordinates are not channel-mapped or
drift-corrected; pixel calibration is an explicit caller assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct

import numpy as np

LAYOUT = "experimental-nim_locb-v5-fixed74-trailer62"
HEADER_BYTES = 25
RECORD_BYTES = 74
TRAILER_RECORD_BYTES = 62

# Offset names intentionally preserve fields whose semantics are unverified.
_FIELDS = [("raw_u32_0", "<u4"), ("frame_index", "<u4"),
           ("channel", "u1"), ("x_raw", "<f4"), ("y_raw", "<f4")]
_FIELDS += [(f"raw_f32_{offset}", "<f4") for offset in range(17, 37, 4)]
_FIELDS += [("raw_flag_37", "u1")]
_FIELDS += [(f"raw_f32_{offset}", "<f4") for offset in range(38, 70, 4)]
_FIELDS += [("raw_u32_70", "<u4")]
RECORD_DTYPE = np.dtype(_FIELDS, align=False)
assert RECORD_DTYPE.itemsize == RECORD_BYTES


@dataclass(frozen=True)
class LocbData:
    path: Path
    records: np.ndarray
    header_raw: bytes
    trailer_raw: bytes
    n_frames: int
    sha256: str

    def metadata(self) -> dict[str, object]:
        channels, counts = np.unique(self.records["channel"], return_counts=True)
        return {
            "source_file": str(self.path), "input_sha256": self.sha256,
            "layout": LAYOUT, "format_version": 5,
            "layout_status": "empirically_inferred_not_vendor_certified",
            "n_records": int(self.records.size), "n_frames": self.n_frames,
            "record_bytes": RECORD_BYTES,
            "channel_counts": {str(int(c)): int(n) for c, n in zip(channels, counts)},
            "coordinate_space": "stored coordinates; no mapping or drift correction applied",
            "coordinate_units": "not encoded by this reader; pixel interpretation must be supplied",
            "trailer_bytes": len(self.trailer_raw),
            "unknown_fields": [name for name in RECORD_DTYPE.names if name.startswith("raw_")],
        }

    def positions_m(self, record_indices: np.ndarray, *, pixel_size_m: float) -> np.ndarray:
        """Convert selected raw coordinates using an explicit pixel hypothesis."""
        if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be positive and finite")
        rows = self.records[record_indices]
        return np.column_stack((rows["frame_index"],
                                rows["x_raw"].astype(float) * pixel_size_m,
                                rows["y_raw"].astype(float) * pixel_size_m))


def read_locb(path: str | Path) -> LocbData:
    """Read only the verified structural layout; reject unsupported variants."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    if len(raw) < HEADER_BYTES or raw[:8] != b"nim_locb":
        raise ValueError("Not a nim_locb file (or truncated header)")
    version = struct.unpack_from("<I", raw, 8)[0]
    if version != 5:
        raise ValueError(f"Unsupported nim_locb version {version}; only observed v5 is supported")
    if raw[12:21] != bytes(9):
        raise ValueError("Unsupported v5 header variant; bytes 12:21 must match the observed layout")
    count = struct.unpack_from("<I", raw, 21)[0]
    end = HEADER_BYTES + RECORD_BYTES * count
    if end + 12 > len(raw):
        raise ValueError("Truncated localization records or unsupported record layout")
    trailer = raw[end:]
    n_frames = struct.unpack_from("<I", trailer, 4)[0]
    if n_frames == 0 or len(trailer) != 12 + TRAILER_RECORD_BYTES * n_frames:
        raise ValueError("Unsupported or truncated v5 trailer; expected 62 bytes per frame")
    if trailer[8:12] != bytes(4):
        raise ValueError("Unsupported v5 trailer header variant")
    trailer_frames = np.ndarray((n_frames,), dtype="<u4", buffer=trailer,
                                offset=12, strides=(TRAILER_RECORD_BYTES,))
    if not np.array_equal(trailer_frames, np.arange(n_frames, dtype=np.uint32)):
        raise ValueError("Trailer frame sequence does not match the observed layout")
    records = np.frombuffer(raw, dtype=RECORD_DTYPE, count=count, offset=HEADER_BYTES)
    if count:
        frames = records["frame_index"]
        if frames.max() >= n_frames:
            raise ValueError("Localization frames are out of range")
        if not np.isin(records["channel"], [0, 1]).all():
            raise ValueError("Unsupported channel field; observed channels are 0 and 1")
        if not (records["raw_flag_37"] == 1).all():
            raise ValueError("Unsupported localization record flag/layout at byte 37")
        for name in RECORD_DTYPE.names:
            if RECORD_DTYPE[name].kind == "f" and not np.isfinite(records[name]).all():
                raise ValueError(f"Non-finite field {name}; file may use a different layout")
    return LocbData(path, records, raw[:HEADER_BYTES], trailer, n_frames,
                    hashlib.sha256(raw).hexdigest())


def load_imagej_roi(path: str | Path) -> np.ndarray:
    """Read an integer polygon/freehand/traced ImageJ ROI as N-by-2 pixels."""
    raw = Path(path).read_bytes()
    if len(raw) < 64 or raw[:4] != b"Iout":
        raise ValueError("Not a supported ImageJ ROI")
    if raw[6] not in (0, 7, 8):
        raise ValueError("Only polygon, freehand, and traced area ROIs are supported")
    options = struct.unpack_from(">H", raw, 50)[0]
    if options & 128:
        raise ValueError("Subpixel ROI coordinates are not supported by this reader")
    top, left = struct.unpack_from(">2H", raw, 8)
    count = struct.unpack_from(">H", raw, 16)[0]
    if count < 3 or len(raw) < 64 + count * 4:
        raise ValueError("Truncated ROI or fewer than three polygon vertices")
    x = np.frombuffer(raw, dtype=">u2", count=count, offset=64).astype(float) + left
    y = np.frombuffer(raw, dtype=">u2", count=count, offset=64 + 2 * count).astype(float) + top
    return np.column_stack((x, y))


def points_in_polygon(xy: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Ray crossing with inclusive boundary, using unshifted pixel coordinates."""
    xy = np.asarray(xy, dtype=float)
    vertices = np.asarray(vertices, dtype=float)
    if (xy.ndim != 2 or xy.shape[1] != 2 or vertices.ndim != 2
            or vertices.shape[1] != 2 or vertices.shape[0] < 3
            or not np.isfinite(xy).all() or not np.isfinite(vertices).all()):
        raise ValueError("Points and polygon must be finite N-by-2 coordinate arrays")
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    candidates = np.flatnonzero(((xy >= lo) & (xy <= hi)).all(axis=1))
    x, y = xy[candidates].T
    inside = np.zeros(candidates.size, dtype=bool)
    boundary = np.zeros_like(inside)
    for first, second in zip(vertices, np.roll(vertices, -1, axis=0)):
        x1, y1 = first
        x2, y2 = second
        dx, dy = x2 - x1, y2 - y1
        cross = (x - x1) * dy - (y - y1) * dx
        # Absolute tolerance is in pixel-distance units after scaling by edge length.
        boundary |= ((np.abs(cross) <= 1e-9 * max(1.0, np.hypot(dx, dy)))
                     & (x >= min(x1, x2) - 1e-9) & (x <= max(x1, x2) + 1e-9)
                     & (y >= min(y1, y2) - 1e-9) & (y <= max(y1, y2) + 1e-9))
        if dy != 0:
            inside ^= ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * dx / dy)
    result = np.zeros(xy.shape[0], dtype=bool)
    result[candidates] = inside | boundary
    return result


def select_records(data: LocbData, *, channel: int, frame_start: int = 0,
                   frame_end: int | None = None, roi: np.ndarray | None = None) -> np.ndarray:
    """Select one channel, inclusive frame bounds, and an optional raw-space ROI."""
    if channel not in (0, 1):
        raise ValueError("channel must be 0 or 1")
    frame_end = data.n_frames - 1 if frame_end is None else frame_end
    if not (0 <= frame_start <= frame_end < data.n_frames):
        raise ValueError("Frame bounds must lie in [0, n_frames - 1]")
    rows = data.records
    indices = np.flatnonzero((rows["channel"] == channel)
                             & (rows["frame_index"] >= frame_start)
                             & (rows["frame_index"] <= frame_end))
    if roi is not None:
        xy = np.column_stack((rows["x_raw"][indices], rows["y_raw"][indices]))
        indices = indices[points_in_polygon(xy, roi)]
    return indices
