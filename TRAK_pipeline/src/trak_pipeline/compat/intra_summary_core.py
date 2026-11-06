"""Read-only, portable BR/condition aggregation of committed signed Intra curves.

This module never reads localizations/images, refits beta, or writes source data.
All three views share a single cell cohort and a single BR cohort.
"""
from __future__ import annotations
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import numpy as np

GROUPS = ('On', 'Not-on')
VIEWS = ('signed', 'area', 'peak')
CONDITIONS = ('WT', 'TDK', 'MDK')
EDGES = np.linspace(0.0, 3.0, 301)
X = (EDGES[:-1] + EDGES[1:]) / 2
WIDTHS = np.diff(EDGES)
SCHEMA_VERSION = 1
SUMMARY_METHOD_VERSION = '2'
AREA_ORDER = 'normalize_cells_then_equal_cell_mean'
CONDITION_AREA_ORDER = 'equal_BR_mean_of_corrected_area_means'


class SummarySourceError(ValueError):
    """A committed result cannot be interpreted under the frozen rules."""


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _natural(value):
    return [int(v) if v.isdigit() else v.lower() for v in re.split(r'(\d+)', str(value))]


def portable_path(data_root, value):
    """Resolve paths recorded on either OS without retaining its drive prefix."""
    root = Path(data_root).resolve()
    native = Path(value)
    if native.is_absolute():
        try:
            native.resolve().relative_to(root)
            return native.resolve()
        except (ValueError, OSError):
            pass
    text = str(value).replace('\\', '/')
    marker = '/' + root.name + '/'
    if marker in text:
        text = text.rsplit(marker, 1)[1]
    elif text.rstrip('/').endswith('/' + root.name):
        text = '.'
    elif text.startswith('/') or re.match(r'^[A-Za-z]:', text):
        raise SummarySourceError('Path is outside the portable data root: ' + str(value))
    path = (root / text).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise SummarySourceError('Path escapes the portable data root: ' + str(value))
    return path


def _relative(root, path):
    return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()


def _reference(root, path):
    data = Path(path).read_bytes()
    return {'relative_path': _relative(root, path), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def build_inventory(control, release):
    control, release = Path(control).resolve(), Path(release).resolve()
    root = control.parent
    source = release / 'expected_inventory.json'
    records = _read_json(source)
    if not isinstance(records, list):
        raise SummarySourceError('Expected inventory must be a list')
    result, seen = [], set()
    for item in records:
        row = dict(item)
        sid = row['sample_id']
        if sid in seen:
            raise SummarySourceError('Duplicate inventory sample ID: ' + sid)
        seen.add(sid)
        if row.get('genotype') not in CONDITIONS:
            raise SummarySourceError('Unexpected condition: ' + str(row.get('genotype')))
        if not row.get('source_directory_relative') and not row.get('source_directory'):
            raise SummarySourceError('Inventory source directory missing: ' + sid)
        directory = portable_path(root, row.get('source_directory_relative') or row.get('source_directory', ''))
        row['source_directory_relative'] = _relative(root, directory)
        row['source_directory'] = str(directory)
        row['data_root'] = str(root)
        result.append(row)
    return sorted(result, key=lambda r: (CONDITIONS.index(r['genotype']), _natural(r['BR']), _natural(r['cell'])))


def evaluate_curves(curves, minimum_area=0.01):
    """Validate one common cohort for all groups/views (not a point-level QC)."""
    reasons = []
    for group in GROUPS:
        if group not in curves:
            reasons.append(group + ':missing_curve')
            continue
        curve = np.asarray(curves[group], dtype=np.float64)
        if curve.shape != (300,):
            reasons.append(group + ':expected_300_bins')
        elif not np.isfinite(curve).all():
            reasons.append(group + ':nonfinite_curve')
        else:
            area = float(np.sum(curve * WIDTHS))
            if abs(area) <= minimum_area:
                reasons.append(group + ':near_zero_signed_area')
            if float(np.max(curve)) <= 0:
                reasons.append(group + ':no_positive_peak')
    return not reasons, reasons


def _true(value):
    return value is True or str(value).strip().lower() == 'true'


def _csv_rows(path, root, refs):
    data = Path(path).read_bytes()
    refs.append({'relative_path': _relative(root, path), 'bytes': len(data),
                 'sha256': hashlib.sha256(data).hexdigest()})
    return list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))


def _check_rules(directory, root, refs):
    path = directory / 'metadata.json'
    if path.is_file():
        data = _read_json(path)
        rules = data.get('config_rules', {})
    else:
        path = directory / 'config.json'
        if not path.is_file():
            raise SummarySourceError('Missing saved TARDIS rule configuration')
        rules = _read_json(path)
    refs.append(_reference(root, path))
    expected = {'frames': 2500, 'tau_frames': 1, 'distance_bins': 300,
                'distance_max_um': 3.0, 'frame_interval_s': 0.02, 'pixel_size_nm': 117.0}
    for key, value in expected.items():
        if key not in rules or not np.isclose(float(rules[key]), value, rtol=0, atol=1e-12):
            raise SummarySourceError('Incompatible saved TARDIS rule: ' + key)
    interval = np.asarray(rules.get('background_fit_um', []), dtype=float)
    if interval.shape != (2,) or not np.allclose(interval, [2.1, 3.0], rtol=0, atol=1e-12):
        raise SummarySourceError('Incompatible beta interval')


def load_cell(entry, expected):
    """Load only committed full-film group curves, validity rows and small rules."""
    if entry.get('_loaded'):
        return entry
    root = Path(entry.get('data_root', expected.get('data_root', '.'))).resolve()
    sid = expected['sample_id']
    result = dict(sample_id=sid, status='complete', eligible=False, curves={}, reasons=[],
                  source_refs=[], output_relative=None, scientific_fingerprint=None, _loaded=True)
    try:
        output = portable_path(root, entry.get('result_directory_relative', entry.get('output', '')))
        result['output_relative'] = _relative(root, output)
        marker_path = output / 'COMPLETED.json'
        marker = _read_json(marker_path)
        if not _true(marker.get('complete')) or marker.get('sample_id') != sid:
            raise SummarySourceError('Missing or mismatched committed completion marker')
        result['scientific_fingerprint'] = marker.get('scientific_fingerprint')
        claimed = entry.get('scientific_fingerprint')
        if claimed and claimed != result['scientific_fingerprint']:
            raise SummarySourceError('Completion registry scientific fingerprint mismatch')
        result['source_refs'].append(_reference(root, marker_path))
        index_path = output / 'RESULT_INDEX.json'
        index = _read_json(index_path) if index_path.is_file() else {}
        if index.get('sample_id', sid) != sid:
            raise SummarySourceError('Saved result index sample ID mismatch')
        if index.get('scientific_fingerprint', result['scientific_fingerprint']) != result['scientific_fingerprint']:
            raise SummarySourceError('Saved result index scientific fingerprint mismatch')
        if index_path.is_file():
            result['source_refs'].append(_reference(root, index_path))
        # MACREF wrappers and original T1 references both resolve to the canonical data.
        source_dirs = index.get('source_directories', {})
        if source_dirs.get('TARDIS_directory'):
            directory = portable_path(root, source_dirs['TARDIS_directory'])
        elif index.get('TARDIS_directory'):
            directory = portable_path(root, index['TARDIS_directory'])
        elif index.get('data'):
            directory = portable_path(root, index['data']) / 'tardis'
        elif index.get('source_result_directory'):
            directory = portable_path(root, index['source_result_directory']) / 'data' / 'tardis'
        else:
            directory = output / 'data' / 'tardis'
        source_directory = expected.get('source_directory_relative') or expected.get('source_directory')
        if source_directory:
            try:
                directory.resolve().relative_to(portable_path(root, source_directory))
            except ValueError:
                raise SummarySourceError('Saved TARDIS source belongs to another cell directory')
        result['tardis_relative'] = _relative(root, directory)
        _check_rules(directory, root, result['source_refs'])
        summary = _csv_rows(directory / 'per_scope_summary.csv', root, result['source_refs'])
        scalar = {}
        for group in GROUPS:
            selected = [row for row in summary if row.get('group') == group and row.get('scope') == 'full']
            if len(selected) != 1:
                raise SummarySourceError(group + ': expected one full-film validity row')
            row = selected[0]
            if int(row['start_frame']) != 0 or int(row['stop_exclusive']) != 2500:
                raise SummarySourceError(group + ': full-film frames are not 0-2499')
            if not _true(row.get('valid')):
                result['reasons'].append(group + ':original_intra_unavailable')
            if not _true(row.get('shape_valid')):
                result['reasons'].append(group + ':original_shape_unavailable')
            if not _true(row.get('tail_supported')):
                result['reasons'].append(group + ':original_tail_support_unavailable')
            beta = float(row['beta'])
            if not np.isfinite(beta):
                result['reasons'].append(group + ':nonfinite_beta')
            scalar[group] = {k: row.get(k) for k in ('valid', 'shape_valid', 'tail_supported', 'beta', 'signed_area', 'shape_status')}
            rows = _csv_rows(directory / 'distribution' / group / 'full.csv', root, result['source_refs'])
            if len(rows) != 300:
                raise SummarySourceError(group + ': expected exactly 300 full-film bins')
            left = np.asarray([float(r['distance_left_um']) for r in rows])
            right = np.asarray([float(r['distance_right_um']) for r in rows])
            if not (np.allclose(left, EDGES[:-1], rtol=0, atol=1e-12) and
                    np.allclose(right, EDGES[1:], rtol=0, atol=1e-12)):
                raise SummarySourceError(group + ': distance bin edges differ from fixed 0-3 um grid')
            if any(r.get('group', group) != group or r.get('scope', 'full') != 'full' or
                   r.get('sample_id', sid) != sid for r in rows):
                raise SummarySourceError(group + ': saved curve identity or scope mismatch')
            curve = np.asarray([float(r['intra_density_um_inv']) for r in rows], dtype=np.float64)
            result['curves'][group] = curve
            saved_area = float(row['signed_area'])
            area = float(np.sum(curve * WIDTHS))
            if np.isfinite(area) and np.isfinite(saved_area) and not np.isclose(area, saved_area, rtol=1e-9, atol=1e-11):
                raise SummarySourceError(group + ': saved signed area differs from curve integral')
        good, reasons = evaluate_curves(result['curves'])
        result['reasons'].extend(reasons)
        result['eligible'] = good and not result['reasons']
        result['scalar'] = scalar
        digest = hashlib.sha256()
        digest.update(str(result['scientific_fingerprint']).encode())
        digest.update(json.dumps(scalar, sort_keys=True).encode())
        for group in GROUPS:
            digest.update(result['curves'][group].astype('<f8').tobytes())
        result['result_digest'] = digest.hexdigest()
    except (OSError, ValueError, TypeError, KeyError) as exc:
        result['status'] = 'invalid_completed'
        result['reasons'].append('committed_result_unreadable: ' + str(exc))
    result['reasons'] = sorted(set(result['reasons']))
    return result


def collect_completed(control, inventory):
    """Merge platform registries by sample ID; conflicting commitments block a BR."""
    control = Path(control).resolve()
    root = control.parent
    expected = {r['sample_id']: r for r in inventory}
    candidates = {sid: [] for sid in expected}
    observed = {sid: [] for sid in expected}
    state_paths = [control / 'state' / 'active_batch.json', control / 'windows' / 'state' / 'active_batch.json']
    winstate = control / 'windows' / 'state'
    if winstate.is_dir():
        state_paths.extend(sorted(winstate.glob('*/active_batch.json')))
    for path in state_paths:
        if not path.is_file():
            continue
        state = _read_json(path)
        for sid, record in state.get('cells', {}).items():
            if sid not in expected:
                continue
            observed[sid].append(record.get('status', 'pending'))
            if record.get('status') in ('complete', 'completed', 'reused'):
                item = dict(record, data_root=str(root), registry_relative=_relative(root, path))
                item.setdefault('scientific_fingerprint', state.get('scientific_fingerprint'))
                candidates[sid].append(item)
    registry_paths = [control / 'windows' / 'portable_completion_registry.json',
                      winstate / 'completed_cells_portable.json']
    if winstate.is_dir():
        registry_paths.extend(sorted(winstate.glob('*/completed_cells_portable.json')))
    for path in registry_paths:
        if not path.is_file():
            continue
        registry = _read_json(path)
        for sid, record in registry.get('entries', {}).items():
            if sid in expected and _true(record.get('complete')):
                item = dict(record, data_root=str(root), registry_relative=_relative(root, path))
                item.setdefault('scientific_fingerprint', registry.get('source_scientific_fingerprint'))
                candidates[sid].append(item)
    result = {}
    for sid, row in expected.items():
        records = candidates[sid]
        if not records:
            failed = any(s in ('failed', 'error', 'invalid', 'invalid_input') for s in observed[sid])
            result[sid] = dict(sample_id=sid, status='failed' if failed else 'pending', eligible=False,
                               reasons=['upstream_cell_failed' if failed else 'not_completed'], curves={},
                               source_refs=[], _loaded=True)
            continue
        loaded, unique_paths = [], set()
        for item in records:
            try:
                path = portable_path(root, item.get('result_directory_relative', item.get('output', '')))
                key = str(path).casefold()
            except SummarySourceError:
                loaded.append(load_cell(item, row))
                continue
            if key in unique_paths:
                continue
            unique_paths.add(key)
            loaded.append(load_cell(item, row))
        digests = {v.get('result_digest') for v in loaded if v.get('status') == 'complete'}
        invalid = [v for v in loaded if v.get('status') != 'complete']
        if len(digests) > 1 or (invalid and len(loaded) > 1):
            result[sid] = dict(sample_id=sid, status='conflict', eligible=False, curves={},
                               reasons=['conflicting_committed_scientific_results'],
                               alternatives=[{k: v.get(k) for k in ('output_relative', 'result_digest', 'status', 'reasons')} for v in loaded],
                               source_refs=[ref for v in loaded for ref in v.get('source_refs', [])], _loaded=True)
        else:
            # Multiple equivalent wrappers are one cell, never multiple equal-weight entries.
            value = sorted(loaded, key=lambda v: str(v.get('output_relative', '')))[0]
            value['equivalent_outputs_relative'] = sorted({v.get('output_relative') for v in loaded if v.get('output_relative')})
            value['registry_sources_relative'] = sorted({i['registry_relative'] for i in records})
            result[sid] = value
    return result


def _empty_view():
    return {'mean': None, 'sd': None, 'sem': None, 'series': {}, 'series_sd': {}, 'series_sem': {}, 'series_n': {}, 'n': 0}


def _mean_view(series, with_sd=False):
    result = _empty_view()
    result['series'] = series
    result['n'] = len(series)
    if series:
        values = np.stack(list(series.values()))
        result['mean'] = np.mean(values, axis=0)
        if with_sd and len(values) > 1:
            result['sd'] = np.std(values, axis=0, ddof=1)
            result['sem'] = result['sd'] / np.sqrt(len(values))
    return result


def _node(level, key, directory, rows, cells, condition=None, BR=None):
    completed = sum(cells[r['sample_id']].get('status') in ('complete', 'invalid_completed', 'conflict') for r in rows)
    failed = sum(cells[r['sample_id']].get('status') == 'failed' for r in rows)
    blocked = any(cells[r['sample_id']].get('status') == 'conflict' for r in rows)
    return dict(level=level, key=key, condition=condition, BR=BR, directory=Path(directory), x=X.copy(), edges=EDGES.copy(),
                progress={'completed': completed, 'expected': len(rows), 'included': 0,
                          'failed': failed, 'terminal': completed + failed, 'status': 'PROGRESS',
                          'valid_BRs': 0, 'expected_BRs': len({(r['genotype'], r['BR']) for r in rows})},
                members=[], exclusions=[], groups={g: {v: _empty_view() for v in VIEWS} for g in GROUPS},
                available=False, blocked=blocked, source_refs=[], signature_payload={})


def _status(node):
    p = node['progress']
    if node['blocked']:
        return 'BLOCKED_SOURCE_CONFLICT'
    if p['terminal'] < p['expected']:
        return 'PROGRESS'
    if p['failed'] or p['included'] < p['completed']:
        return 'COMPLETE_WITH_FAILURES_OR_EXCLUSIONS'
    return 'COMPLETE'


def _finalize(node):
    node['progress']['status'] = _status(node)
    # All saved identities use relative paths and stable scientific values only.
    node['signature_payload'] = {
        'schema_version': SCHEMA_VERSION, 'aggregation_method': {'method_id': 'hierarchy_summary', 'method_version': SUMMARY_METHOD_VERSION, 'area_order': AREA_ORDER, 'condition_area_order': CONDITION_AREA_ORDER}, 'key': node['key'], 'progress': node['progress'],
        'members': node['members'], 'exclusions': node['exclusions'],
        'sources': sorted(node['source_refs'], key=lambda r: (r['relative_path'], r.get('sha256', ''))),
        'blocked': node['blocked'],
    }
    return node


def summarize_cells(inventory, cells, config=None):
    """Pure two-level equal-weight aggregation; no point/pair-count weighting."""
    config = config or {}
    root = Path(config.get('data_root', inventory[0].get('data_root', '.') if inventory else '.'))
    cells = {sid: dict(value) for sid, value in cells.items()}
    for row in inventory:
        sid = row['sample_id']
        cell = cells.setdefault(sid, dict(sample_id=sid, status='pending', eligible=False, curves={}, reasons=['not_completed']))
        if cell.get('status') == 'complete':
            good, reasons = evaluate_curves(cell.get('curves', {}))
            cell['reasons'] = sorted(set(cell.get('reasons', []) + reasons))
            cell['eligible'] = bool(good and cell.get('eligible', True) and not cell['reasons'])
    declared_replicas = config.get('replicas')
    if declared_replicas is not None:
        if (not isinstance(declared_replicas, list) or not declared_replicas or
                any(not isinstance(br, str) or not re.fullmatch(r'BR[1-9][0-9]*', br) for br in declared_replicas) or
                len(set(declared_replicas)) != len(declared_replicas)):
            raise SummarySourceError('Declared replicas must be unique explicit BR labels')
    br_nodes, condition_nodes = [], []
    for condition in CONDITIONS:
        crows = [r for r in inventory if r['genotype'] == condition]
        present = {r['BR'] for r in crows}
        if declared_replicas is not None and crows and present != set(declared_replicas):
            raise SummarySourceError('Expected inventory replica roster differs for ' + condition +
                                     ': declared=' + str(declared_replicas) + '; observed=' + str(sorted(present, key=_natural)))
        replica_order = declared_replicas if declared_replicas is not None else sorted(present, key=_natural)
        for BR in (br for br in replica_order if br in present):
            rows = [r for r in crows if r['BR'] == BR]
            first_dir = Path(rows[0]['source_directory_relative'])
            node = _node('BR', condition + '/' + BR, root / first_dir.parent, rows, cells, condition, BR)
            selected = []
            for row in rows:
                cell = cells[row['sample_id']]
                meta = {k: row.get(k) for k in ('sample_id', 'genotype', 'BR', 'cell', 'molecule', 'acquisition_date', 'source_directory_relative')}
                meta.update(status=cell.get('status'), output_relative=cell.get('output_relative'), result_digest=cell.get('result_digest'))
                node['source_refs'].extend(cell.get('source_refs', []))
                if cell.get('eligible') and cell.get('status') == 'complete':
                    node['members'].append(meta)
                    selected.append((row['sample_id'], cell))
                else:
                    node['exclusions'].append(dict(meta, reasons=cell.get('reasons', ['not_eligible'])))
            node['progress']['included'] = len(selected)
            if selected and not node['blocked']:
                for group in GROUPS:
                    raw = {sid: np.asarray(cell['curves'][group], dtype=np.float64) for sid, cell in selected}
                    node['groups'][group]['signed'] = _mean_view(raw)
                    # Give every eligible cell one equal weight in the area-shape view.
                    area_series = {sid: curve / np.sum(curve * WIDTHS) for sid, curve in raw.items()}
                    node['groups'][group]['area'] = _mean_view(area_series)
                    node['groups'][group]['peak'] = _mean_view({sid: curve / np.max(curve) for sid, curve in raw.items()})
                canonical_curves = {g: node['groups'][g]['signed']['mean'] for g in GROUPS}
                node['canonical_eligible'], node['canonical_exclusion_reasons'] = evaluate_curves(canonical_curves)
                node['available'] = True
            else:
                node['canonical_eligible'] = False
                node['canonical_exclusion_reasons'] = ['source_conflict' if node['blocked'] else 'no_eligible_cells']
            node['progress']['valid_BRs'] = int(node['canonical_eligible'])
            br_nodes.append(_finalize(node))
        node = _node('condition', condition, root / condition, crows, cells, condition)
        children = [n for n in br_nodes if n['condition'] == condition]
        eligible_brs = [n for n in children if n['canonical_eligible'] and not n['blocked']]
        node['blocked'] = any(n['blocked'] for n in children)
        node['members'] = [{'BR': n['BR'], 'key': n['key'], 'included_cells': n['progress']['included']} for n in eligible_brs]
        node['exclusions'] = [{'BR': n['BR'], 'key': n['key'], 'reasons': n['canonical_exclusion_reasons']} for n in children if n not in eligible_brs]
        node['source_refs'] = [ref for n in children for ref in n['source_refs']]
        node['progress']['included'] = sum(n['progress']['included'] for n in eligible_brs)
        node['progress']['valid_BRs'] = len(eligible_brs)
        if eligible_brs and not node['blocked']:
            for group in GROUPS:
                raw = {n['BR']: n['groups'][group]['signed']['mean'] for n in eligible_brs}
                node['groups'][group]['signed'] = _mean_view(raw, True)
                node['groups'][group]['area'] = _mean_view({child['BR']: child['groups'][group]['area']['mean'] for child in eligible_brs}, True)
                node['groups'][group]['peak'] = _mean_view({BR: curve / np.max(curve) for BR, curve in raw.items()}, True)
            node['available'] = True
        condition_nodes.append(_finalize(node))
    node = _node('comparison', 'WT_TDK_MDK', root, inventory, cells)
    eligible_conditions = [n for n in condition_nodes if n['available'] and not n['blocked']]
    node['members'] = [{'condition': n['condition'], 'valid_BRs': n['progress']['valid_BRs'], 'included_cells': n['progress']['included']} for n in eligible_conditions]
    node['exclusions'] = [{'condition': n['condition'], 'reasons': ['blocked_source_conflict' if n['blocked'] else 'no_valid_BR']} for n in condition_nodes if n not in eligible_conditions]
    node['blocked'] = False  # A conflicting condition is explicitly omitted, never substituted.
    node['progress']['included'] = sum(n['progress']['included'] for n in eligible_conditions)
    node['progress']['valid_BRs'] = sum(n['progress']['valid_BRs'] for n in eligible_conditions)
    node['source_refs'] = [ref for n in condition_nodes for ref in n['source_refs']]
    for group in GROUPS:
        for view in VIEWS:
            node['groups'][group][view]['series'] = {n['condition']: n['groups'][group][view]['mean'] for n in eligible_conditions}
            node['groups'][group][view]['series_sd'] = {n['condition']: n['groups'][group][view]['sd'] for n in eligible_conditions}
            node['groups'][group][view]['series_sem'] = {n['condition']: n['groups'][group][view]['sem'] for n in eligible_conditions}
            node['groups'][group][view]['series_n'] = {n['condition']: n['groups'][group][view]['n'] for n in eligible_conditions}
            node['groups'][group][view]['n'] = len(eligible_conditions)
    node['available'] = bool(eligible_conditions)
    return br_nodes + condition_nodes + [_finalize(node)]


def build_summaries(control, release, config=None):
    inventory = build_inventory(control, release)
    cells = collect_completed(control, inventory)
    settings = dict(config or {})
    molecule = settings.get('molecule')
    if molecule and any(row.get('molecule') != molecule for row in inventory):
        raise SummarySourceError('Inventory molecule differs from the summary configuration')
    definition = Path(control) / 'scientific_definition.json'
    expected_fingerprint = settings.get('scientific_fingerprint')
    if not expected_fingerprint and definition.is_file():
        expected_fingerprint = _read_json(definition).get('fingerprint')
    if expected_fingerprint:
        for cell in cells.values():
            if cell.get('status') == 'complete' and cell.get('scientific_fingerprint') != expected_fingerprint:
                cell['eligible'] = False
                cell['reasons'] = sorted(set(cell.get('reasons', []) + ['incompatible_scientific_fingerprint']))
    settings['data_root'] = str(Path(control).resolve().parent)
    nodes = summarize_cells(inventory, cells, settings)
    return {'nodes': nodes, 'inventory': inventory, 'cells': cells,
            'discovery': {'expected_cells': len(inventory),
                          'completed_cells': sum(c['status'] in ('complete', 'invalid_completed', 'conflict') for c in cells.values()),
                          'eligible_cells': sum(bool(c.get('eligible')) for c in cells.values()),
                          'conflicts': [sid for sid, c in cells.items() if c['status'] == 'conflict'],
                          'rule': 'Only committed cell results; shared On/Not-on cohort; equal cell and BR weights'}}
