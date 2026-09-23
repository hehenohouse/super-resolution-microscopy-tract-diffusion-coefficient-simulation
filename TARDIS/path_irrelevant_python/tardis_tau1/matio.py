from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Iterator

import numpy as np

_MI_MATRIX = 14
_MI_COMPRESSED = 15
_NUMERIC_DTYPES = {
    1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4",
    7: "f4", 9: "f8", 12: "i8", 13: "u8",
}


def load_mat_variable(path: str | Path, variable_name: str = "pos") -> np.ndarray:
    """Load one real numeric matrix from a MATLAB v5/v7 (non-HDF5) MAT file.

    This small reader avoids a SciPy dependency and supports the numeric matrix
    layout used by the localization files. MATLAB v7.3/HDF5 files require scipy
    or h5py and are rejected with a clear message.
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw.startswith(b"\x89HDF"):
        raise ValueError("MATLAB v7.3/HDF5 input is not supported without h5py")
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
