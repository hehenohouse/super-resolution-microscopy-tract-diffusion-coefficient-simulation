#!/usr/bin/env python3
"""Independent, read-only-by-default audit of a completed ROI-only fit cache.

No detection, fitting, masking, tracking or TARDIS is performed.  Writing
ancillary audit records to the cache requires the explicit --write-report
switch.  The runner's original validation, completion marker, scientific
tables, source snapshots, and logs are never changed by this tool.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pandas as pd
import tifffile

PROJECT = Path(__file__).resolve().parents[1]
N_FRAMES = 2500


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def fingerprint(path):
    path = Path(path).absolute()
    sha = digest(path)
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def read_csv(path):
    return pd.read_csv(path, float_precision='round_trip', low_memory=False)


def bool_values(series):
    """Refuse missing/ambiguous booleans instead of treating strings as true."""
    require(not series.isna().any(), f'Missing boolean: {series.name}')
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    values = series.astype(str).str.lower()
    require(values.isin(['true', 'false', '0', '1']).all(), f'Invalid boolean: {series.name}')
    return values.isin(['true', '1']).to_numpy()


def same_columns(left, right, exclude=()):
    """Exact values, including NaNs; CSV inferred integer/float dtype may differ."""
    fields = [name for name in left.columns if name not in exclude]
    require(set(fields).issubset(right.columns), 'Required original columns missing')
    for name in fields:
        pd.testing.assert_series_equal(left[name].reset_index(drop=True),
            right[name].reset_index(drop=True), check_dtype=False, check_exact=True,
            obj=f'Preserved column {name}')
    return fields


def load_roi(config, out):
    path = PROJECT / 'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py'
    spec = importlib.util.spec_from_file_location('roi_sample_audit_locbio', path)
    library = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = library
    spec.loader.exec_module(library)
    vertices = library.load_imagej_roi(config['roi_file'])
    found = []
    with h5py.File(config['mat_file'], 'r') as handle:
        for reference in handle['data'][()].ravel(order='F'):
            if not reference:
                continue
            record = handle[reference]
            if not isinstance(record, h5py.Group) or 'name' not in record:
                continue
            name = ''.join(chr(int(v)) for v in record['name'][()].ravel(order='F') if int(v))
            if name == config['mat_record']:
                found.append(record['roiData/roiCoordinates'][()].T.astype(float))
    require(len(found) == 1, 'Expected one MAT record')
    np.testing.assert_array_equal(vertices, found[0])
    np.testing.assert_array_equal(vertices, read_csv(out / 'roi_vertices.csv').to_numpy())
    require(len(vertices) == config['expected_roi_vertices'], 'ROI vertex count differs')
    yy, xx = np.mgrid[:684, :428]
    roi = library.points_in_polygon(np.c_[xx.ravel(), yy.ravel()], vertices).reshape(684, 428)
    np.testing.assert_array_equal(roi, np.load(out / 'roi_mask.npy', allow_pickle=False))
    require(int(roi.sum()) == config['expected_roi_pixels'], 'ROI pixel count differs')
    return library, vertices, roi


def check_qc(fits, vertices, library, config):
    finite = np.isfinite(fits[['fit_x_px', 'fit_y_px']]).all(axis=1).to_numpy()
    inside = np.zeros(len(fits), bool)
    inside[finite] = library.points_in_polygon(fits.loc[finite, ['fit_x_px', 'fit_y_px']].to_numpy(), vertices)
    b = {name: bool_values(fits[name]) for name in ['fit_attempted', 'fit_success', 'active_bound',
        'fit_usable_interior', 'finite_uncertainty', 'suspect_overlap']}
    np.testing.assert_array_equal(bool_values(fits.fitted_center_inside_ROI), inside)
    eligible = b['fit_usable_interior'] & b['finite_uncertainty'] & inside
    selected = eligible & ~b['suspect_overlap']
    np.testing.assert_array_equal(bool_values(fits.histogram_eligible), eligible)
    np.testing.assert_array_equal(bool_values(fits.qc_unflagged_model_fit), selected)
    reason = np.select([~b['fit_attempted'], ~b['fit_success'], b['active_bound'],
        ~b['finite_uncertainty'], ~inside, b['suspect_overlap']],
        ['unavailable_support', 'nonconverged', 'active_parameter_bound', 'unavailable_covariance',
         'refined_center_outside_ROI', 'overlap_review'], default='unflagged_model_fit')
    np.testing.assert_array_equal(fits.audit_status, reason)
    require(np.all(b['fit_attempted'][selected] & b['fit_success'][selected] & ~b['active_bound'][selected]),
        'QC-selected point has failed or bound-active fit')
    clean = fits.loc[selected]
    fields = ['fit_x_px', 'fit_y_px', 'x_se_nm', 'y_se_nm', 'photons_e', 'bg_e_per_pixel', 'sx_px', 'sy_px']
    require(np.isfinite(clean[fields]).all().all(), 'QC-selected model parameters not finite')
    require(clean[['x_se_nm', 'y_se_nm', 'photons_e']].gt(0).all().all(), 'Invalid QC-selected SE or signal')
    require(bool_values(clean.background_supported).all(), 'Selected fit lacks annular support')
    require(clean.background_pixels.ge(config['minimum_background_pixels']).all(), 'Selected fit has insufficient annular pixels')
    lo, hi = config['width_bounds_px']
    require(((clean[['sx_px', 'sy_px']] > lo) & (clean[['sx_px', 'sy_px']] < hi)).all().all(),
        'QC-selected width not inside recorded limits')
    np.testing.assert_array_equal(fits.time_s, fits.frame * config['frame_interval_s'])
    for axis in ['x', 'y']:
        np.testing.assert_array_equal(fits[f'lp{axis}_model_nm'], fits[f'{axis}_se_nm'])
        np.testing.assert_array_equal(fits[f'lp{axis}_model_px'], fits[f'{axis}_se_nm'] / config['pixel_size_nm'])
    require(fits.calibration_status.eq(config['calibration_status']).all(), 'Calibration status differs')
    require(not bool_values(fits.quantitatively_calibrated).any(), 'Nominal results labelled calibrated')
    return selected


def audit(out):
    complete = read_json(out / 'COMPLETED.json')
    require(complete.get('delivery_checks_passed') is True, 'Full runner is not complete')
    require(read_json(out / 'run_status.json').get('status') == 'complete', 'Runner is still active/incomplete')
    c = read_json(out / 'config.json')
    require('mask_directory' not in c, 'This auditor is for ROI-only localization caches')
    require(not c['tracking'] and not c['tardis'], 'Unexpected enabled downstream analysis')
    require(c['pixel_size_nm'] == 117 and c['frame_interval_s'] == .02, 'Unexpected physical scale')
    source = read_json(out / 'source_validation.json')
    require(source['sample_id'] == c['sample_id'] and source['sample_name'] == c['sample_name'], 'Sample identity differs')
    require(source['ROI_MAT_vertices_exact'] and source['no_coordinate_transform'], 'Original ROI contract differs')
    require(source.get('mitochondrial_mask_required') is False and source.get('mitochondrial_mask_verified') is False,
        'ROI-only localization must not claim mask verification')
    before = read_json(out / 'input_fingerprints_before.json')
    require(before == read_json(out / 'input_fingerprints_after.json'), 'Recorded source fingerprints changed')
    require(len({r['path'] for r in before}) == len(before), 'Duplicate source fingerprint')
    print('AUDIT current source content hashes', flush=True)
    for row in before:
        require(fingerprint(row['path']) == row, f"Current source changed: {row['path']}")
    source_paths = {r['path'] for r in before}
    require(all(str(Path(c[k]).absolute()) in source_paths for k in ['C1_file', 'C2_file', 'roi_file', 'mat_file']),
        'Required source not fingerprinted')
    manifest = read_json(out / 'output_fingerprints.json')
    print('AUDIT existing output manifest', len(manifest), flush=True)
    require(len({r['path'] for r in manifest}) == len(manifest), 'Duplicate output manifest path')
    for row in manifest:
        require(Path(row['path']).is_relative_to(out), 'Manifest references a file outside this cache')
        require(fingerprint(row['path']) == row, f"Cached output changed: {row['path']}")
    expected_manifest = {str(path) for path in out.rglob('*') if path.is_file()
        and path.name not in ['output_fingerprints.json', 'COMPLETED.json', 'run.log']}
    require({r['path'] for r in manifest} == expected_manifest, 'Output manifest coverage differs')
    library, vertices, roi = load_roi(c, out)
    fits = read_csv(out / 'all_ROI_target_fits.csv')
    detections = read_csv(out / 'all_detections.csv')
    frames = read_csv(out / 'frame_counts.csv')
    background = read_csv(out / 'annular_background_all_ROI_targets.csv')
    pilot = read_csv(out / 'pilot/all_ROI_target_fits.csv')
    summary = read_json(out / 'summary.json')
    np.testing.assert_array_equal(frames.frame, np.arange(N_FRAMES))
    np.testing.assert_array_equal(frames.time_s, frames.frame * c['frame_interval_s'])
    for table in [detections, fits, background, pilot]:
        require(table.candidate_index.is_unique and table.source_detection_uid.is_unique, 'Candidate identity not unique')
        require(table.frame.between(0, N_FRAMES - 1).all(), 'Invalid original frame')
        require(np.all(np.diff(table.frame.to_numpy()) >= 0), 'Rows not in original frame order')
    np.testing.assert_array_equal(detections.candidate_index, np.arange(len(detections)))
    chunk = (detections.frame.to_numpy(int) // c['native_detection_chunk_frames']) * c['native_detection_chunk_frames']
    expected_uids = [f"{c['sample_id']}:frames{a:04d}_{a+24:04d}:{int(s)}" for a, s in zip(chunk, detections.spot_id)]
    np.testing.assert_array_equal(detections.source_detection_uid, expected_uids)
    raw_inside = library.points_in_polygon(detections[['x_px', 'y_px']].to_numpy(), vertices)
    np.testing.assert_array_equal(bool_values(detections.roi_inside), raw_inside)
    targets = detections.loc[raw_inside]
    for name in ['candidate_index', 'spot_id', 'frame', 'source_detection_uid', 'source_detection_cache']:
        np.testing.assert_array_equal(targets[name], fits[name])
    for name in ['candidate_index', 'frame', 'source_detection_uid']:
        np.testing.assert_array_equal(background[name], fits[name])
    selected = check_qc(fits, vertices, library, c)
    matched = fits.set_index('source_detection_uid', drop=False).loc[pilot.source_detection_uid].reset_index(drop=True)
    pilot_fields = same_columns(pilot, matched, {'candidate_index', 'pilot_candidate_index', 'reused_from_pilot'})
    require(bool_values(matched.reused_from_pilot).all(), 'Pilot fits not marked reused')
    np.testing.assert_array_equal(matched.pilot_candidate_index, pilot.candidate_index)
    expected_reuse = fits.source_detection_uid.isin(pilot.source_detection_uid).to_numpy()
    np.testing.assert_array_equal(bool_values(fits.reused_from_pilot), expected_reuse)
    pilot_frames = [f for a, b in c['pilot_segments'] for f in range(a, b + 1)]
    require(len(pilot_frames) == 150 and set(pilot.frame).issubset(pilot_frames), 'Pilot frame selection differs')
    np.testing.assert_array_equal(read_csv(out / 'pilot/frame_counts.csv').frame, pilot_frames)
    require(set(pilot.source_detection_uid) == set(fits.loc[fits.frame.isin(pilot_frames), 'source_detection_uid']),
        'Pilot/full ROI membership differs')
    # Every frame, including an empty detection/ROI frame, remains represented.
    fg = {int(k): v for k, v in fits.groupby('frame', sort=False)}
    bg = {int(k): v for k, v in background.groupby('frame', sort=False)}
    dg = {int(k): v for k, v in detections.groupby('frame', sort=False)}
    for row in frames.itertuples():
        f = int(row.frame)
        local = fg.get(f, fits.iloc[:0])
        require(row.FOV_candidates == len(dg.get(f, ())) and row.ROI_candidates == len(local), f'Frame {f} count mismatch')
        require(row.selected == int(bool_values(local.qc_unflagged_model_fit).sum()), f'Frame {f} selection mismatch')
        require(row.selected + row.QC_excluded == row.ROI_candidates, f'Frame {f} conservation failed')
        reasons = local.audit_status.value_counts()
        for column in frames.columns:
            if column.startswith('QC_reason__'):
                require(frames.loc[f, column] == reasons.get(column[len('QC_reason__'):], 0), f'Frame {f} QC reason mismatch')
    for key, expected in [('frames', N_FRAMES), ('FOV_candidates', len(detections)),
        ('ROI_candidates', len(fits)), ('selected', int(selected.sum())), ('QC_excluded', int((~selected).sum()))]:
        require(summary[key] == expected, f'Summary {key} differs')
        require(complete['counts'][key] == expected, f'Completion {key} differs')
    require(summary['sample_name'] == c['sample_name'], 'Summary sample differs')
    # This runner's summary counts reuse executed in its last invocation, not
    # persistent reuse across a resumed invocation. Row flags/UIDs above are
    # the authoritative check that every pilot fit is preserved.
    require(0 <= summary['reused_pilot_fits'] <= len(pilot), 'Invalid invocation reuse count')
    require(summary['QC_reasons'] == {str(k): int(v) for k, v in fits.audit_status.value_counts().items()}, 'Summary reasons differ')
    for end in range(250, N_FRAMES + 1, 250):
        checkpoint = read_json(out / 'checkpoints' / f'fits_{end:04d}.json')
        require(checkpoint['completed_frames'] == end, 'Checkpoint frame range differs')
        for key in ['ROI_candidates', 'selected', 'QC_excluded']:
            require(checkpoint[key] == int(frames.loc[frames.frame < end, key].sum()), f'Checkpoint {end} {key} differs')
    require(len(list((out / 'checkpoints').glob('fits_*.json'))) == 10, 'Expected exactly ten full checkpoints')
    math = read_json(out / 'math_validation.json')
    require(math and all(v is True for v in math.values()), 'Recorded mathematical checks did not pass')
    page_manifest = read_json(out / 'source_page_manifest.json')
    require([r['frame'] for r in page_manifest] == list(range(N_FRAMES)), 'Original page manifest differs')
    page_hashes = {r['frame']: r['pixel_sha256'] for r in page_manifest}
    native_hashes = {}
    print('AUDIT native detection provenance', flush=True)
    for start in range(0, N_FRAMES, 25):
        tag = f'frames{start:04d}_{start+24:04d}'
        info = read_json(out / 'native_detection' / (tag + '.json'))
        require(info['start'] == start and info['stop_exclusive'] == start + 25, 'Native chunk range differs')
        for key, expected in [('trackmate_version', c['trackmate_version']), ('radius_um', c['radius_um']),
            ('threshold', c['threshold']), ('median_filter', c['median_filter']), ('subpixel', c['subpixel_localization']),
            ('tracking', False), ('pixel_roundtrip_exact', True)]:
            require(info[key] == expected, f'Native setting {key} differs')
        native_path = out / 'native_detection' / (tag + '.csv')
        require(digest(native_path) == info['csv_sha256'], 'Native detection CSV changed')
        table = read_csv(native_path)
        native_selected = detections.loc[detections.frame.between(start, start + 24)].reset_index(drop=True)
        same_columns(table, native_selected)
        require(info['rows'] == len(table), 'Native row count differs')
        require([p['frame'] for p in info['page_hashes']] == list(range(start, start + 25)), 'Native page mapping differs')
        native_hashes.update({p['frame']: p['pixel_sha256'] for p in info['page_hashes']})
    require(native_hashes == page_hashes, 'Native/source page hashes differ')
    for key in ['C1_file', 'C2_file']:
        with tifffile.TiffFile(c[key]) as handle:
            require(len(handle.pages) == N_FRAMES and handle.series[0].axes == 'TYX', f'{key} time axis differs')
            require(handle.series[0].shape == (N_FRAMES, 684, 428), f'{key} shape differs')
            require(all(page.dtype == np.dtype('uint16') and page.shape == (684, 428) for page in handle.pages),
                f'{key} page dimensions/dtype differ')
    print('AUDIT 2500 original C2 pages and per-frame cache identities', flush=True)
    with tifffile.TiffFile(c['C2_file']) as handle:
        for f, page in enumerate(handle.pages):
            sha = hashlib.sha256(np.ascontiguousarray(page.asarray()).tobytes()).hexdigest()
            require(sha == page_hashes[f], f'Raw C2 page {f} changed')
            marker = read_json(out / 'frame_fits' / f'frame_{f:04d}.json')
            local = fg.get(f, fits.iloc[:0])
            require(marker['frame'] == f and marker['source_pixel_sha256'] == sha, f'Frame {f} marker source differs')
            require(marker['UIDs'] == local.source_detection_uid.tolist() and marker['rows'] == len(local), f'Frame {f} identity differs')
            for relative, field in [(f'frame_{f:04d}.csv', 'fit_sha256'), (f'background_{f:04d}.csv', 'background_sha256'),
                (f'annular_pixels_{f:04d}.npz', 'pixels_sha256')]:
                require(digest(out / 'frame_fits' / relative) == marker[field], f'Frame {f} cache changed: {relative}')
            same_columns(read_csv(out / 'frame_fits' / f'frame_{f:04d}.csv'), local)
            local_bg = bg.get(f, background.iloc[:0])
            same_columns(read_csv(out / 'frame_fits' / f'background_{f:04d}.csv'), local_bg)
            with np.load(out / 'frame_fits' / f'annular_pixels_{f:04d}.npz', allow_pickle=False) as pixels:
                np.testing.assert_array_equal(pixels['candidate_index'], local.candidate_index)
                np.testing.assert_array_equal(pixels['source_detection_uid'], local.source_detection_uid)
                require(pixels['offsets'][0] == 0 and pixels['offsets'][-1] == len(pixels['xy_pixels']), 'Annular offsets invalid')
                require(len(pixels['offsets']) == len(local) + 1 and np.all(np.diff(pixels['offsets']) >= 0), 'Annular rows invalid')
                xy = pixels['xy_pixels']
                require(xy.ndim == 2 and xy.shape[1] == 2 and np.issubdtype(xy.dtype, np.integer), 'Annular XY shape/type invalid')
                require(np.all((xy[:, 0] >= 0) & (xy[:, 0] < 428) & (xy[:, 1] >= 0) & (xy[:, 1] < 684)), 'Annular pixel outside FOV')
                require(roi[xy[:, 1], xy[:, 0]].all(), 'Annular pixel outside original ROI')
                if f in pilot_frames:
                    with np.load(out / 'pilot' / f'annular_pixels_{f:04d}.npz', allow_pickle=False) as original:
                        for field in ['xy_pixels', 'offsets', 'source_detection_uid']:
                            np.testing.assert_array_equal(pixels[field], original[field])
                        np.testing.assert_array_equal(original['candidate_index'], local.pilot_candidate_index)
            if f in pilot_frames:
                original_bg = read_csv(out / 'pilot' / f'background_{f:04d}.csv')
                same_columns(original_bg, local_bg, {'candidate_index'})
            if (f + 1) % 250 == 0:
                print('AUDIT_FRAMES', f + 1, flush=True)
    # Recheck protected inputs and already-verified cache metadata after audit.
    for row in before:
        require(fingerprint(row['path']) == row, f"Source changed while auditing: {row['path']}")
    for row in manifest:
        stat = Path(row['path']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (row['bytes'], row['mtime_ns']), 'Cache changed while auditing')
    blocks = []
    for start in range(0, N_FRAMES, 250):
        local = fits.loc[fits.frame.between(start, start + 249)]
        clean = local.loc[bool_values(local.qc_unflagged_model_fit)]
        record = dict(first_frame=start, last_frame=start + 249, ROI_candidates=len(local), selected=len(clean),
            QC_excluded=len(local) - len(clean))
        for key in ['x_se_nm', 'y_se_nm', 'photons_e', 'bg_e_per_pixel', 'sx_px', 'sy_px']:
            values = clean[key].to_numpy()
            for label, p in [('q25', 25), ('median', 50), ('q75', 75), ('p95', 95)]:
                record[key + '_' + label] = float(np.percentile(values, p)) if len(values) else None
        blocks.append(record)
    report = dict(all_checks_passed=True, audit_utc=datetime.now(timezone.utc).isoformat(),
        sample_id=c['sample_id'], sample_name=c['sample_name'], frames=N_FRAMES,
        counts={k: summary[k] for k in ['FOV_candidates', 'ROI_candidates', 'selected', 'QC_excluded']},
        ROI_vertices=len(vertices), ROI_pixels=int(roi.sum()), pilot_frames=pilot_frames, pilot_rows_reused=len(pilot),
        runner_last_invocation_reuse_count=summary['reused_pilot_fits'],
        pilot_original_fields_exactly_equal=pilot_fields, pilot_reindex_only=True,
        stable_UIDs_and_global_indices_verified=True,
        QC_membership_recomputed_from_preserved_model_flags_and_original_ROI=True,
        Hessian_not_recomputed_and_no_refitting=True,
        all_2500_original_C2_page_hashes_verified=True, all_per_frame_cache_and_annular_identities_verified=True,
        ten_250_frame_checkpoints_conserved=True, current_source_SHA_unchanged=True,
        existing_output_manifest_verified_files=len(manifest), original_runner_records_unchanged=True,
        mitochondrial_mask_required=False, mitochondrial_mask_used=False,
        calibration_status=c['calibration_status'], model_SE_coverage_validated=False,
        auditor_fingerprint=fingerprint(__file__))
    return report, blocks, c, before, manifest


def write_report(out, report, blocks, c, sources, manifest, command):
    """Only ancillary new outputs; do not mutate original runner conclusions."""
    names = ['additional_validation.json', 'per_250_frame_fit_summary.csv', 'field_units.json', 'README.txt',
        'audit_software_versions.json', 'audit_requirements.lock.txt', 'audit_detector_requirements.lock.txt',
        'validation_command.txt', 'dependency_fingerprints.json', 'runner_output_fingerprints.json', 'dependency_snapshot']
    require(not any((out / name).exists() for name in names), 'Refusing to overwrite an earlier audit; inspect it first')
    packages = {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'scipy', 'h5py', 'tifffile', 'Pillow']}
    freeze = subprocess.run([sys.executable, '-B', '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
    detector = Path(c['detector_python'])
    if not detector.is_absolute():
        detector = (PROJECT / detector).absolute()
    detector_freeze = subprocess.run([str(detector), '-B', '-m', 'pip', 'freeze'], capture_output=True, text=True, check=True).stdout
    dependencies = []
    for row in sources + [fingerprint(__file__)]:
        path = Path(row['path'])
        if path.is_relative_to(PROJECT):
            require(fingerprint(path) == row, f'Source changed before snapshot: {path}')
            target = out / 'dependency_snapshot' / path.relative_to(PROJECT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            require(digest(target) == row['sha256'], 'Snapshot hash mismatch')
            dependencies.append(dict(source=row, snapshot=str(target), snapshot_sha256=digest(target)))
    units = dict(frame='Original zero-based frame, 0..2499', time_s=f"Seconds = frame * {c['frame_interval_s']}",
        source_detection_uid='sample_id:25-frame-source-cache-tag:TrackMate spot_id; stable across pilot/full',
        candidate_index='Full-FOV detection row in original frame order; not a track identifier',
        fit_x_px='Refined center X, native C2 pixels', fit_y_px='Refined center Y, native C2 pixels',
        photons_e='Nominal whole-Gaussian photoelectron equivalents, including PSF extrapolation outside fit box',
        signal_inside_box_e=f"Nominal photoelectron equivalents inside {c['box_px']}x{c['box_px']} fit box",
        bg_e_per_pixel='Jointly fitted background mean, nominal electron equivalents/pixel',
        sx_px='Gaussian standard deviation X, native pixels', sy_px='Gaussian standard deviation Y, native pixels',
        x_se_nm='Full observed-Hessian local X SE including nuisance covariance, nm; coverage unvalidated',
        y_se_nm='Full observed-Hessian local Y SE including nuisance covariance, nm; coverage unvalidated',
        lpx_model_px=f"x_se_nm / {c['pixel_size_nm']}, native pixels; nominal model SE",
        lpy_model_px=f"y_se_nm / {c['pixel_size_nm']}, native pixels; nominal model SE",
        N_se_e='Whole-Gaussian signal SE, nominal electron equivalents',
        bg_se_e_per_pixel='Background mean SE, nominal electron equivalents/pixel',
        background_variance_e2='Empirical annular intensity variance, nominal electron-equivalents squared',
        background_constraint_se_e='External annular background-mean constraint SE, nominal electron equivalents/pixel')
    dump(out / 'additional_validation.json', report)
    pd.DataFrame(blocks).to_csv(out / 'per_250_frame_fit_summary.csv', index=False)
    dump(out / 'field_units.json', units)
    dump(out / 'dependency_fingerprints.json', dependencies)
    dump(out / 'audit_software_versions.json', dict(python=sys.version, executable=sys.executable, packages=packages,
        original_runner_versions_file='software_versions.json', audit_environment_is_not_a_new_fit_run=True))
    (out / 'audit_requirements.lock.txt').write_text(freeze)
    (out / 'audit_detector_requirements.lock.txt').write_text(detector_freeze)
    (out / 'validation_command.txt').write_text(command + '\n')
    camera = c['camera']
    (out / 'README.txt').write_text(
        f"{c['sample_name']}: original C2 frames 0..2499; {c['pixel_size_nm']} nm/pixel; {c['frame_interval_s']} s/frame.\n"
        f"TrackMate {c['trackmate_version']} native LoG radius {c['radius_um']} um, threshold {c['threshold']}, "
        f"median3x3={c['median_filter']}, subpixel={c['subpixel_localization']}. No tracking.\n"
        f"Pixel-integrated Gaussian {c['box_px']}x{c['box_px']}; annulus {c['annulus_inner_radius_px']}<r<="
        f"{c['annulus_outer_radius_px']} px, exclude all-FOV candidate cores <= {c['neighbor_core_radius_px']} px; "
        f"original ROI and nonzero pixels only, >= {c['minimum_background_pixels']} pixels.\n"
        'Background mean constraint variance=max(sample variance/n, eight-sector delete-one jackknife variance).\n'
        'Full observed-Hessian local SE includes nuisance covariance; empirical background variance is fixed.\n'
        f"Nominal offset {camera['baseline_adu']} ADU, sensitivity {camera['sensitivity_e_per_adu']} e/ADU, "
        f"EM gain {camera['em_gain']}, read noise {camera['read_noise_e_rms']} e RMS.\n"
        'Signal and standard errors use manufacturer-nominal calibration, not measured photons or validated precision coverage.\n'
        'Original QC: converged interior fit, finite positive covariance, refined center inside original cell ROI, no residual-overlap flag.\n'
        'All FOV detections and ROI fit failures retained; no SE-magnitude cutoff.\n'
        f"Full counts: {report['counts']}; pilot ROI rows reused exactly: {report['pilot_rows_reused']}.\n"
        'UIDs identify source detections; full candidate indices are frame-order rows, not tracks or independent molecules.\n'
        'No mitochondrial mask is required, used or verified by localization or this audit.\n'
        'No TARDIS, trajectory linking, segmentation, biological-state fitting or causal inference was performed.\n'
        'additional_validation.json is an independent audit; original runner validation, completion and logs are unchanged.\n')
    dump(out / 'runner_output_fingerprints.json', manifest)
    # Scientific outputs were verified by SHA before writing. Reuse records only
    # after checking their unchanged metadata; new ancillary records are hashed.
    old = {r['path']: r for r in manifest}
    for row in manifest:
        stat = Path(row['path']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (row['bytes'], row['mtime_ns']), 'Protected cache output changed')
    updated = []
    for path in sorted(out.rglob('*')):
        if path.is_file() and path.name not in ['output_fingerprints.json', 'COMPLETED.json', 'run.log']:
            updated.append(old[str(path)] if str(path) in old else fingerprint(path))
    dump(out / 'output_fingerprints.json', updated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='Completed ROI-only localization cache')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true', help='Default: no cache writes')
    mode.add_argument('--write-report', action='store_true', help='Explicitly add ancillary audit records to this cache')
    parser.add_argument('--report', type=Path, help='Optional workspace-only JSON report (also on failed checks)')
    args = parser.parse_args()
    out = args.output.absolute()
    if args.report is not None:
        target = args.report.resolve()
        require(target.is_relative_to(PROJECT) and not target.is_relative_to(out.resolve()), '--report must be inside the workspace, outside the audited cache')
        require(target.parent.is_dir(), '--report parent must already exist')
    try:
        report, blocks, config, sources, manifest = audit(out)
        report['mode'] = 'write_report' if args.write_report else 'check_only'
        if args.write_report:
            write_report(out, report, blocks, config, sources, manifest,
                shlex.join([sys.executable, '-B', str(Path(__file__).absolute()), *sys.argv[1:]]))
        if args.report is not None:
            dump(args.report, dict(validation=report, per_250_frame_summary=blocks))
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    except Exception as error:
        if args.report is not None:
            dump(args.report, dict(all_checks_passed=False, error_type=type(error).__name__, error=str(error),
                output=str(out), audit_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
