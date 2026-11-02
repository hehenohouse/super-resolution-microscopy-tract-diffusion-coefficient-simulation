from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

_MI_MATRIX = 14
_MI_COMPRESSED = 15
_NUMERIC_DTYPES = {
    1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4",
    7: "f4", 9: "f8", 12: "i8", 13: "u8",
}
_FILTERED_SUFFIXES = ("_SpotsMaskFiltered", "_TracksFiltered")


@dataclass
class V73BaseRecord:
    """One decoded unfiltered cell record from a MATLAB v7.3 file."""

    name: str
    record_index: int
    positions: np.ndarray
    metadata: dict[str, Any]


def load_mat_variable(path: str | Path, variable_name: str = "pos") -> np.ndarray:
    """Load one real numeric matrix from a MATLAB v5/v7 (non-HDF5) MAT file."""
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(b"\x89HDF") or b"MATLAB 7.3 MAT-file" in raw[:128]:
        raise ValueError(
            "MATLAB v7.3/HDF5 input requires load_v73_cell_record() and h5py"
        )
    if len(raw) < 128 or b"MATLAB 5.0 MAT-file" not in raw[:116]:
        raise ValueError(f"Not a MATLAB v5 MAT file: {path}")
    endian_marker = raw[126:128]
    if endian_marker == b"IM":
        endian = "<"
    elif endian_marker == b"MI":
        endian = ">"
    else:
        raise ValueError("Unrecognized MAT-file endian marker")
    for element_type, payload in _elements(raw, 128, endian):
        matrices = _matrix_payloads(element_type, payload, endian)
        for matrix_payload in matrices:
            name, array = _decode_matrix(matrix_payload, endian)
            if name == variable_name:
                return array
    raise KeyError(f'Variable "{variable_name}" not found in {path}')


def load_v73_base_records(
    path: str | Path,
    *,
    frame_interval_s: float,
    pixel_size_m: float,
    variable_name: str = "data",
) -> tuple[list[V73BaseRecord], list[str], list[dict[str, Any]]]:
    """Decode every Base record once, preserving MATLAB cell-array order."""
    _validate_v73_calibration(frame_interval_s, pixel_size_m)
    h5py = _import_h5py()
    path = Path(path)
    records: list[V73BaseRecord] = []
    discovered: list[str] = []
    failures: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        references = _v73_references(handle, path, variable_name, h5py)
        candidates: list[tuple[int, str, Any]] = []
        for index, reference in enumerate(references):
            if not reference:
                continue
            record = handle[reference]
            if not isinstance(record, h5py.Group) or "name" not in record:
                continue
            name = _decode_matlab_char(record["name"][()])
            if not name or name.endswith(_FILTERED_SUFFIXES):
                continue
            discovered.append(name)
            candidates.append((index, name, record))

        name_counts = {name: discovered.count(name) for name in set(discovered)}
        for index, name, record in candidates:
            if name_counts[name] > 1:
                failures.append(
                    {
                        "cell_id": name,
                        "record_index": index,
                        "error": f'Base record "{name}" occurs more than once',
                    }
                )
                continue
            try:
                positions, metadata = _decode_v73_record(
                    record, name, index, frame_interval_s, pixel_size_m
                )
            except (KeyError, TypeError, ValueError) as exc:
                failures.append(
                    {"cell_id": name, "record_index": index, "error": str(exc)}
                )
                continue
            records.append(V73BaseRecord(name, index, positions, metadata))
    return records, discovered, failures


def load_v73_cell_record(
    path: str | Path,
    record_name: str,
    *,
    frame_interval_s: float,
    pixel_size_m: float,
    variable_name: str = "data",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one Base cell record as ``[integer frame, x_m, y_m]``."""
    if not record_name or record_name.endswith(_FILTERED_SUFFIXES):
        raise ValueError("record_name must identify an unfiltered Base record")
    _validate_v73_calibration(frame_interval_s, pixel_size_m)
    h5py = _import_h5py()
    path = Path(path)
    matches: list[tuple[int, Any]] = []
    with h5py.File(path, "r") as handle:
        references = _v73_references(handle, path, variable_name, h5py)
        for index, reference in enumerate(references):
            if not reference:
                continue
            record = handle[reference]
            if not isinstance(record, h5py.Group) or "name" not in record:
                continue
            if _decode_matlab_char(record["name"][()]) == record_name:
                matches.append((index, record))
        if not matches:
            raise KeyError(f'Base record "{record_name}" not found in {path}')
        if len(matches) > 1:
            raise ValueError(f'Base record "{record_name}" occurs more than once')
        index, record = matches[0]
        return _decode_v73_record(
            record, record_name, index, frame_interval_s, pixel_size_m
        )


def _decode_v73_record(
    record: Any,
    record_name: str,
    record_index: int,
    frame_interval_s: float,
    pixel_size_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    required = ("time", "x_data", "y_data")
    missing = [field for field in required if field not in record]
    if missing:
        raise ValueError(
            f'Record "{record_name}" is missing fields: {", ".join(missing)}'
        )
    time_s = _numeric_vector(record["time"][()], "time")
    x_pixel = _numeric_vector(record["x_data"][()], "x_data")
    y_pixel = _numeric_vector(record["y_data"][()], "y_data")
    if not (time_s.size == x_pixel.size == y_pixel.size):
        raise ValueError("time, x_data, and y_data must have identical lengths")
    if time_s.size == 0:
        raise ValueError(f'Record "{record_name}" contains no localizations')

    stored_interval = None
    metadata_group = record.get("tracksMetaData")
    if metadata_group is not None and "frameInterval" in metadata_group:
        values = _numeric_vector(metadata_group["frameInterval"][()], "frameInterval")
        if values.size != 1:
            raise ValueError("tracksMetaData.frameInterval must be scalar")
        stored_interval = float(values[0])
        if not np.isclose(stored_interval, frame_interval_s, rtol=0, atol=1e-12):
            raise ValueError(
                "Configured frame interval does not match tracksMetaData: "
                f"{frame_interval_s} s versus {stored_interval} s"
            )

    frame_float = time_s / frame_interval_s
    frames = np.rint(frame_float)
    if not np.allclose(frame_float, frames, rtol=0, atol=1e-8):
        error = float(np.max(np.abs(frame_float - frames)))
        raise ValueError(f"time/frame_interval_s is not integral; max error={error:g}")
    unique_time = np.unique(time_s)
    if unique_time.size > 1:
        steps = np.diff(unique_time)
        if not np.allclose(steps, frame_interval_s, rtol=0, atol=1e-10):
            raise ValueError("Unique time points are not spaced by frame_interval_s")

    positions = np.column_stack(
        (frames.astype(np.int64), x_pixel * pixel_size_m, y_pixel * pixel_size_m)
    )
    metadata = {
        "record_name": record_name,
        "record_index": int(record_index),
        "record_variant": "base",
        "input_coordinate_unit": "pixel",
        "pixel_size_m": float(pixel_size_m),
        "input_time_unit": "s",
        "frame_interval_s": float(frame_interval_s),
        "stored_frame_interval_s": stored_interval,
        "time_min_s": float(time_s.min()),
        "time_max_s": float(time_s.max()),
        "analysis_coordinate_unit": "m",
    }
    return positions, metadata


def _validate_v73_calibration(frame_interval_s: float, pixel_size_m: float) -> None:
    if not np.isfinite(frame_interval_s) or frame_interval_s <= 0:
        raise ValueError("frame_interval_s must be a positive finite SI value")
    if not np.isfinite(pixel_size_m) or pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be a positive finite SI value")


def _import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "MATLAB v7.3 input requires h5py; install the project dependencies"
        ) from exc
    return h5py


def _v73_references(handle: Any, path: Path, variable_name: str, h5py: Any) -> np.ndarray:
    if variable_name not in handle:
        raise KeyError(f'Variable "{variable_name}" not found in {path}')
    data = handle[variable_name]
    if h5py.check_dtype(ref=data.dtype) is None:
        raise ValueError(f'Variable "{variable_name}" is not a MATLAB cell array')
    return np.asarray(data[()]).reshape(-1, order="F")


def _decode_matlab_char(value: np.ndarray) -> str:
    codes = np.asarray(value).reshape(-1, order="F")
    return "".join(chr(int(code)) for code in codes if int(code) != 0)


def _numeric_vector(value: np.ndarray, field: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{field} must be numeric")
    vector = array.astype(float, copy=False).reshape(-1, order="F")
    if not np.isfinite(vector).all():
        raise ValueError(f"{field} must contain only finite values")
    return vector


def _matrix_payloads(element_type: int, payload: bytes, endian: str) -> Iterator[bytes]:
    if element_type == _MI_MATRIX:
        yield payload
    elif element_type == _MI_COMPRESSED:
        inflated = zlib.decompress(payload)
        for nested_type, nested_payload in _elements(inflated, 0, endian):
            yield from _matrix_payloads(nested_type, nested_payload, endian)


def _elements(blob: bytes, offset: int, endian: str) -> Iterator[tuple[int, bytes]]:
    length = len(blob)
    while offset + 8 <= length:
        first, second = struct.unpack_from(endian + "II", blob, offset)
        small_bytes = first >> 16
        if small_bytes:
            element_type = first & 0xFFFF
            yield element_type, blob[offset + 4 : offset + 4 + small_bytes]
            offset += 8
            continue
        element_type = first
        byte_count = second
        start = offset + 8
        end = start + byte_count
        if end > length:
            raise ValueError("Truncated MAT data element")
        yield element_type, blob[start:end]
        offset = end + ((8 - byte_count % 8) % 8)


def _decode_matrix(payload: bytes, endian: str) -> tuple[str, np.ndarray]:
    parts = list(_elements(payload, 0, endian))
    if len(parts) < 4:
        raise ValueError("Incomplete miMATRIX element")
    _, dimensions_raw = parts[1]
    dimensions = np.frombuffer(dimensions_raw, dtype=np.dtype(endian + "i4")).astype(int)
    dimensions = tuple(int(value) for value in dimensions if value > 0)
    _, name_raw = parts[2]
    name = name_raw.decode("utf-8", errors="replace").rstrip("\x00")
    real_type, real_raw = parts[3]
    if real_type not in _NUMERIC_DTYPES:
        raise ValueError(f"Unsupported MAT numeric type: {real_type}")
    dtype = np.dtype(endian + _NUMERIC_DTYPES[real_type])
    values = np.frombuffer(real_raw, dtype=dtype)
    expected = int(np.prod(dimensions))
    if values.size < expected:
        raise ValueError(f"Matrix {name!r} has truncated real data")
    array = values[:expected].reshape(dimensions, order="F")
    return name, np.asarray(array)
