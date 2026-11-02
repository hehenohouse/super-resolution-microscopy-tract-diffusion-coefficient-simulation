"""Explicit scientific dependency graph; styling never invalidates science."""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy
from .identity import scientific_key
from .cache import CacheStore
from .types import Artifact

@dataclass(frozen=True)
class StageSpec:
    stage: str
    method_id: str
    method_version: str
    dependencies: tuple[str, ...]
    scientific: bool = True

# C1 and C2 are deliberately separated. No ROI enters full-image Nellie.
DEPENDENCIES = {
    'detection': ('C2',),
    'fit': ('detection', 'C2', 'roi'),
    'broad': ('C1',),
    'support_counts': ('broad',),
    'mask_threshold': ('support_counts',),
    'mask_cleanup': ('mask_threshold',),
    'mask_roi': ('mask_cleanup', 'roi'),
    'assignment': ('fit', 'mask_cleanup', 'roi', 'C1'),
    'distances': ('fit', 'assignment'),
    'decomposition': ('distances',),
    'bootstrap': ('distances', 'decomposition'),
    'c1_mean': ('C1',),
    'statistics': ('fit', 'assignment', 'mask_roi', 'mask_cleanup', 'roi', 'support_counts', 'decomposition', 'bootstrap'),
    'presentation': ('C1', 'C2', 'roi', 'detection', 'fit', 'mask_cleanup', 'mask_roi', 'assignment',
                     'decomposition', 'bootstrap', 'c1_mean', 'statistics'),
}


def stage_parameters(config, spec):
    raw = deepcopy(config.get('methods', {}).get(spec.stage, {}))
    if 'parameters' in raw:
        params = raw['parameters']
    else:
        params = {k: v for k, v in raw.items() if k not in ('method_id', 'method_version', 'version')}
    if not isinstance(params, dict):
        raise ValueError('Stage scientific parameters must be a mapping: ' + spec.stage)
    # Acquisition values that affect geometry/time belong to explicit stage
    # parameters supplied by config defaults; no global config hash is used.
    if spec.stage == 'presentation':
        params = dict(params, style=deepcopy(config.get('style', {})))
    if spec.stage == 'distances' and int(params.get('tau_frames', 1)) != 1:
        raise ValueError('Distance method v1 only supports tau_frames=1; a new implementation/method version is required')
    return params


def default_specs(config=None):
    config = config or {}
    specs = []
    for stage, deps in DEPENDENCIES.items():
        m = config.get('methods', {}).get(stage, {})
        specs.append(StageSpec(stage, m.get('method_id', stage), str(m.get('method_version', m.get('version', '1'))),
                               tuple(deps), stage != 'presentation'))
    return specs


def expected_identity(spec, parameters, inputs):
    return scientific_key(spec.method_id, spec.method_version, parameters, {d: inputs[d].id for d in spec.dependencies})


def build_plan(config, sample, existing_artifacts=None):
    """Read-only plan, including pending IDs so invalidation propagates precisely."""
    existing = dict(existing_artifacts or {})
    available = {s: a for s, a in existing.items() if s in ('C1', 'C2', 'roi')}
    cache = CacheStore(config, sample)
    result = []
    for spec in default_specs(config):
        params = stage_parameters(config, spec)
        absent = [d for d in spec.dependencies if d not in available]
        if absent:
            result.append({'stage': spec.stage, 'scientific': spec.scientific, 'action': 'blocked',
                           'reason': 'missing_inputs', 'missing': absent, 'artifact_id': None})
            continue
        inputs = {d: available[d] for d in spec.dependencies}
        key = expected_identity(spec, params, inputs)
        legacy = existing.get(spec.stage)
        match = cache.lookup(spec, params, inputs) or (legacy if legacy is not None and legacy.id == key else None)
        if match is not None:
            action = 'reuse'
            reason = 'compatible_legacy_logical_artifact_not_materialized'if match.metadata.get('materialization_required')else'compatible_saved_artifact'
            available[spec.stage] = match
        else:
            action = 'compute' if spec.scientific else 'publish'
            reason = 'changed_parameters_or_upstream' if legacy is not None else cache.last_miss_reason or 'missing_saved_artifact'
            # A planned artifact's identity is known without running it. This
            # allows a dry run to distinguish unaffected from dependent stages.
            available[spec.stage] = Artifact(key, spec.stage, {}, {'planned_only': True}, params,
                                             {d: inputs[d].id for d in spec.dependencies}, spec.method_id, spec.method_version)
        result.append({'stage': spec.stage, 'scientific': spec.scientific, 'action': action, 'reason': reason,
                       'artifact_id': key, 'dependencies': list(spec.dependencies), 'parameters': params,
                       'virtual': bool(match is not None and match.metadata.get('virtual'))})
    return result
