"""Scientific identity deliberately excludes source code, presentation, and locations."""
from __future__ import annotations
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

class IdentityError(ValueError):
    pass

def normalize(value):
    """Canonical JSON primitives; reject ambiguous/nonfinite scientific parameters."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IdentityError('Nonfinite scientific parameter')
        if value == 0:
            return 0
        if value.is_integer():
            return int(value)
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise IdentityError('Scientific parameter keys must be strings')
        return {k: normalize(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    # Explicit scalar conversion supports NumPy without importing it here.
    if hasattr(value, 'item') and not isinstance(value, Path):
        return normalize(value.item())
    raise IdentityError('Unsupported scientific parameter type: ' + type(value).__name__)

def canonical_json(value):
    return json.dumps(normalize(value), sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()

def scientific_key(method_id, method_version, parameters, upstream_ids):
    if not isinstance(method_id, str) or not method_id or not str(method_version):
        raise IdentityError('Explicit nonempty method ID and version are required')
    if any(not isinstance(v, str) or not v for v in upstream_ids.values()):
        raise IdentityError('Every upstream artifact needs a scientific ID')
    return digest({'schema': 1, 'method_id': method_id, 'method_version': str(method_version),
                   'parameters': parameters, 'upstream_ids': upstream_ids})
