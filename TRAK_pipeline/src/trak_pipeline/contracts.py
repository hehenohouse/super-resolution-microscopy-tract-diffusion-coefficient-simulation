"""Spatial/time contracts, UID validation, and explicit missing-data semantics."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Any

class ContractError(ValueError):
    pass

PIXEL_CENTERS = 'integer_centers_xy;closed_unit_squares'
TEN_FRAMES = (0, 49, 200, 625, 1200, 1250, 1875, 2000, 2450, 2499)


def _integer(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value:
        raise ContractError(field + ' must be an integer')
    return int(value)


def validate_mask_contract(value: dict[str, Any], expected_shape=None,
                           expected_pixel_size_nm=None, expected_frame_interval_s=None):
    """Validate metadata without reading an entire movie or inventing registration.

    frame_to_block may be a list aligned with original_frame_ids or a mapping.
    Missing frames remain absent/-1. block_frame_ids records actual participants;
    expected_block_frame_ids may additionally record each complete requested block.
    An incomplete block is unknown, never an all-zero negative observation.
    """
    c = dict(value)
    if c.get('axes') != 'TYX':
        raise ContractError('Mask axes must explicitly be TYX')
    shape = [_integer(x, 'shape') for x in c.get('shape', ())]
    if len(shape) != 3 or min(shape) < 1:
        raise ContractError('Mask shape must be positive TYX')
    if expected_shape is not None and tuple(shape) != tuple(expected_shape):
        raise ContractError('Mask shape disagrees with its declared spatial/time contract')
    for name, expected in [('pixel_size_nm', expected_pixel_size_nm), ('frame_interval_s', expected_frame_interval_s)]:
        v = c.get(name)
        if not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
            raise ContractError(name + ' must be explicitly positive')
        if expected is not None and not math.isclose(v, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ContractError(name + ' conflicts with acquisition')
    offset = c.get('crop_offset_xy_px')
    if offset is None or len(offset) != 2 or any(not math.isfinite(float(x)) for x in offset):
        raise ContractError('Explicit crop_offset_xy_px is required')
    convention = c.get('pixel_center_convention')
    if not isinstance(convention, str) or not convention:
        raise ContractError('Explicit pixel center convention is required')
    if not any(s in convention.lower() for s in ['integer', 'center', 'centre']):
        raise ContractError('Unrecognized pixel center convention')
    registration = c.get('registration')
    if isinstance(registration, dict):
        transform = registration.get('transform', c.get('coordinate_transform'))
        status = registration.get('status')
    else:
        transform = c.get('coordinate_transform', c.get('transform'))
        status = c.get('registration_status', registration)
    declared_transform = c.get('coordinate_transform', c.get('transform'))
    if declared_transform is not None and declared_transform not in ('identity', 'identity_assumption'):
        raise ContractError('Coordinate transform contradicts declared identity registration')
    if transform not in ('identity', 'identity_assumption'):
        raise ContractError('Only explicitly declared identity C1-C2 registration is implemented')
    if status not in ('unverified', 'identity/unverified', 'exploratory_identity_registration_unverified', 'verified'):
        raise ContractError('Explicit C1-C2 registration verification status is required')
    frames = [_integer(f, 'original_frame_ids') for f in c.get('original_frame_ids', [])]
    if not frames or min(frames) < 0 or len(set(frames)) != len(frames):
        raise ContractError('Original frame IDs must be unique, nonnegative, and explicit')
    raw = c.get('frame_to_block')
    if isinstance(raw, list):
        if len(raw) != len(frames):
            raise ContractError('frame_to_block list must match original_frame_ids')
        mapping = {str(f): _integer(b, 'frame_to_block') for f, b in zip(frames, raw)}
    elif isinstance(raw, dict):
        mapping = {str(_integer(int(f), 'frame_to_block frame')): _integer(b, 'frame_to_block block') for f, b in raw.items()}
        if set(mapping) - set(map(str, frames)):
            raise ContractError('frame_to_block contains undeclared original frames')
    else:
        raise ContractError('Explicit original-frame to mask-block mapping is required')
    if any(b < -1 or b >= shape[0] for b in mapping.values()):
        raise ContractError('frame_to_block references unavailable mask blocks')
    block_frames = c.get('block_frame_ids')
    if isinstance(block_frames, dict):
        block_frames = [block_frames.get(str(i), block_frames.get(i, [])) for i in range(shape[0])]
    if not isinstance(block_frames, list) or len(block_frames) != shape[0]:
        raise ContractError('Actual participating frames are required for each support block')
    block_frames = [[_integer(f, 'block_frame_ids') for f in fs] for fs in block_frames]
    valid = c.get('block_valid')
    valid = list(valid) if isinstance(valid, list) else valid
    reasons = list(c.get('block_reasons', [''] * shape[0]))
    if not isinstance(valid, list) or len(valid) != shape[0] or len(reasons) != shape[0]:
        raise ContractError('Explicit block validity and matching reasons are required')
    if any(type(v) is not bool for v in valid):
        raise ContractError('block_valid must contain booleans')
    expected = c.get('expected_block_frame_ids', block_frames)
    if len(expected) != shape[0]:
        raise ContractError('Expected support participants do not match block count')
    expected_seen = set()
    for requested in expected:
        if len(set(requested)) != len(requested) or expected_seen.intersection(requested):
            raise ContractError('Requested support windows overlap or repeat a frame')
        expected_seen.update(requested)
    seen = set()
    for i, fs in enumerate(block_frames):
        if len(set(fs)) != len(fs) or seen.intersection(fs):
            raise ContractError('A frame occurs multiple times in support participants')
        seen.update(fs)
        if any(str(f) not in mapping or mapping[str(f)] != i for f in fs):
            raise ContractError('Support participants disagree with original-frame mapping')
        incomplete = set(fs) != set(expected[i]) or not fs
        if incomplete:
            valid[i] = False
            reasons[i] = reasons[i] or 'incomplete_support_block'
        if not valid[i] and not reasons[i]:
            reasons[i] = 'invalid_support_block'
    c.update(shape=shape, original_frame_ids=frames, frame_to_block=mapping,
             block_frame_ids=block_frames, block_valid=list(valid), block_reasons=list(reasons),
             crop_offset_xy_px=list(offset), registration={'transform': 'identity', 'status': status})
    return c


def unknown_reason_for_frame(contract, frame):
    c = contract
    raw = c.get('frame_to_block', {})
    if isinstance(raw, list):
        try:
            b = raw[c['original_frame_ids'].index(int(frame))]
        except (ValueError, IndexError):
            return 'missing_original_frame'
    else:
        b = raw.get(str(int(frame)), raw.get(int(frame), -1))
    if b is None or b < 0:
        return 'missing_original_frame'
    if b >= len(c.get('block_valid', [])):
        return 'missing_mask_block'
    if not c['block_valid'][b]:
        return c.get('block_reasons', [''] * len(c['block_valid']))[b] or 'invalid_support_block'
    if int(frame) not in c['block_frame_ids'][b]:
        return 'missing_original_frame'
    return ''


def stable_uid(detection_source_identity, native_detection_id):
    if not isinstance(detection_source_identity, str) or not detection_source_identity:
        raise ContractError('A persisted detection source identity is required')
    if native_detection_id is None or str(native_detection_id) == '':
        raise ContractError('A native detection ID is required')
    return detection_source_identity + ':' + str(native_detection_id)


def validate_localization_table(rows, detection_rows=None, selected_field='spatial_selected', frame_interval_s=0.02):
    """Check IDs/SE meanings and ten-frame correspondence; never infer missing SEs."""
    seen = set()
    sample = []
    selected = 0
    required = ('source_detection_uid', 'spot_id', 'frame', 'fit_x_px', 'fit_y_px', 'x_se_nm', 'y_se_nm',
                'uncertainty_status', 'qc_unflagged_model_fit')
    for row in rows:
        missing = [k for k in required if k not in row]
        if missing:
            raise ContractError('Legacy localization fields missing: ' + ', '.join(missing))
        uid = str(row['source_detection_uid'])
        if not uid or uid in seen:
            raise ContractError('Localization detection UID is empty or duplicated: ' + uid)
        seen.add(uid)
        if not uid.endswith(':' + str(row['spot_id'])):
            raise ContractError('Persistent UID does not identify the native detection ID: ' + uid)
        if row.get('candidate_uid') not in (None, '', uid):
            raise ContractError('Candidate UID disagrees with detection UID: ' + uid)
        frame = int(row['frame'])
        if frame < 0 or float(row['frame']) != frame:
            raise ContractError('Localization frame must be a nonnegative original integer')
        if 'time_s' in row and row['time_s'] not in ('', None) and not math.isclose(float(row['time_s']), frame * frame_interval_s, abs_tol=1e-9):
            raise ContractError('Original frame/time mapping changed: ' + uid)
        use = str(row.get(selected_field, row['qc_unflagged_model_fit'])).lower() in ('true', '1')
        if use:
            selected += 1
            if str(row['qc_unflagged_model_fit']).lower() not in ('true', '1'):
                raise ContractError('Selected localization fails original QC: ' + uid)
            for field in ['fit_x_px', 'fit_y_px', 'x_se_nm', 'y_se_nm']:
                try:
                    value = float(row[field])
                except (ValueError, TypeError):
                    raise ContractError('Selected localization lacks finite ' + field + ': ' + uid)
                if not math.isfinite(value) or ('_se_' in field and value <= 0):
                    raise ContractError('Invalid selected localization ' + field + ': ' + uid)
            if 'full_observed_Hessian' not in str(row['uncertainty_status']):
                raise ContractError('Unrecognized standard-error meaning: ' + uid)
        if frame in TEN_FRAMES:
            sample.append(row)
    checked = 0
    if detection_rows is not None:
        wanted = {str(r['source_detection_uid']): r for r in sample}
        found = set()
        detector_seen = set()
        for r in detection_rows:
            uid = str(r.get('source_detection_uid', ''))
            if not uid or uid in detector_seen:
                raise ContractError('Detection UID is empty or duplicated: ' + uid)
            detector_seen.add(uid)
            if uid in wanted:
                fit = wanted[uid]
                if int(r['frame']) != int(fit['frame']):
                    raise ContractError('Detection and fit original frame disagree: ' + uid)
                for field in ('x_px', 'y_px'):
                    if field in fit and (field not in r or not math.isclose(float(r[field]), float(fit[field]), abs_tol=1e-10)):
                        raise ContractError('Detection correspondence changed: ' + uid)
                found.add(uid)
        if found != set(wanted):
            raise ContractError('Missing ten-frame localization/detection UID correspondence')
        if not seen.issubset(detector_seen):
            raise ContractError('Localization UID absent from detection source')
        checked = len(found)
    return {'rows': len(seen), 'selected': selected, 'unique_UIDs': True,
            'ten_frame_detection_correspondences': checked, 'checked_frames': list(TEN_FRAMES),
            'original_detector_coordinates_duplicated_in_fit_table': all(k in row for row in sample for k in ('x_px','y_px')),
            'missing_optional_original_detection_columns': sorted({k for row in sample for k in ('x_px','y_px') if k not in row}),
            'standard_errors': 'full observed Hessian including nuisance parameters; nominal calibration'}
