#!/usr/bin/env python3
"""Lean TARDIS orchestration around the frozen numerical cores.

Raw counts, original resampling IDs/weights, scalar draws, per-field validity,
statuses and pointwise bands are retained. Density draws are computed once in
memory and discarded after summaries; no formal hypothesis test is performed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

RELEASE = Path(__file__).resolve().parent
FIELDS = ('total', 'background', 'inter', 'intra', 'shape')
METRICS = ('beta', 'signed_area', 'main_mass', 'tail_mass', 'tail_main_ratio')


def require(value, message):
    if not value:
        raise ValueError(message)


def configure(job):
    from batch_common import configure_environment
    configure_environment(job['runtime_dir'])
    for relative in ['picasso_pilot', 'picasso_full_sensitivity', 'picasso_roi_sensitivity',
                     'TARDIS/path_irrelevant_python', 'mito_tardis']:
        sys.path.insert(0, str(RELEASE / 'vendor' / relative))


def load_science():
    global np, pd, mc, rc
    import numpy as np
    import pandas as pd
    import mito_core as mc
    import roi_core as rc


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes(order='C')).hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    if isinstance(value, float) and not __import__('math').isfinite(value):
        return None
    return value


def dump(path, value):
    # Shared guarded UTF-8 write closes the temporary file before Windows replace.
    from batch_common import dump as portable_dump
    portable_dump(path, jsonable(value))


def identity(job):
    return {k: job.get(k) for k in ('sample_id', 'run_id', 'molecule', 'genotype', 'BR', 'cell', 'acquisition_date')}


def tagged(frame, job):
    frame = frame.copy()
    for k, v in reversed(list(identity(job).items())):
        if k not in frame:
            frame.insert(0, k, v)
    return frame


def boolean(series):
    text = series.astype(str).str.lower()
    require(text.isin(['true', 'false', '1', '0']).all(), 'Unrecognized boolean value')
    return text.isin(['true', '1'])


def selected_table(audit, c):
    require(audit.candidate_index.is_unique, 'Candidate IDs must be unique')
    keep = boolean(audit.qc_unflagged_model_fit) & boolean(audit.fitted_center_inside_ROI)
    if 'spatial_selected' in audit:
        np.testing.assert_array_equal(keep, boolean(audit.spatial_selected))
    label = 'pointwise_label' if 'pointwise_label' in audit else 'postmin32_pointwise_label'
    require(label in audit, 'Missing frozen pointwise labels')
    table = audit.loc[keep].copy()
    table['tardis_group'] = table[label].map({**mc.LABELS, 'On': 'On', 'Not-on': 'Not-on'})
    require(table.tardis_group.notna().all(), 'Selected label is not On/Not-on/unknown')
    numeric = ['candidate_index', 'frame', 'time_s', 'fit_x_px', 'fit_y_px', 'x_se_nm', 'y_se_nm',
               'photons_e', 'bg_e_per_pixel', 'sx_px', 'sy_px']
    for key in numeric:
        table[key] = table[key].map(int if key in ('candidate_index', 'frame') else float)
    require(np.isfinite(table[numeric].to_numpy(float)).all(), 'Selected numerical values must be finite')
    require(table.frame.between(0, c['frames'] - 1).all(), 'Frame outside original movie')
    np.testing.assert_allclose(table.time_s, table.frame * c['frame_interval_s'], atol=1e-12, rtol=0)
    require(table[['x_se_nm', 'y_se_nm']].gt(0).all().all(), 'Selected SE must be strictly positive')
    return table, keep


def curve_effects(observed, c):
    width = c['distance_max_um'] / c['distance_bins']
    mid = (np.arange(c['distance_bins']) + .5) * width
    result = []
    for scope, start, stop in mc.scopes(c):
        a, b = observed['On', scope], observed['Not-on', scope]
        difference = (a['shape'][0] - b['shape'][0]) / width
        available = bool(np.isfinite(difference).all())
        result.append(dict(scope=scope, shape_difference_available=available,
            max_abs_density_difference=float(np.max(abs(difference))) if available else np.nan,
            max_abs_difference_distance_um=float(mid[np.argmax(abs(difference))]) if available else np.nan,
            integrated_abs_signed_density_difference=float(abs(difference).sum() * width) if available else np.nan,
            normalized_main_signed_area_difference=float(difference[mid < .1].sum() * width) if available else np.nan,
            normalized_tail_signed_area_difference=float(difference[(mid >= .1) & (mid < .25)].sum() * width) if available else np.nan))
    return result


def field_status(d, field):
    finite = np.isfinite(d[field]).all(1)
    status = np.full(len(finite), 'available_'+field, dtype='<U64')
    if field == 'shape':
        return np.asarray(d['shape_status'], dtype='<U64')
    if field in ('total', 'background'):
        status[~finite] = 'unavailable_empty_'+field
    else:
        status[~finite & ~d['tail_supported']] = 'unavailable_sparse_tail'
        status[~finite & d['tail_supported']] = 'unavailable_invalid_distribution'
    return status


def scalar_status(d, key):
    invalid = ~np.isfinite(d[key])
    status = np.full(len(invalid), 'available', dtype='<U64')
    status[invalid & ~d['tail_supported']] = 'unavailable_sparse_tail'
    status[invalid & d['tail_supported']] = 'unavailable_invalid_distribution'
    if key == 'tail_main_ratio':
        status[invalid & d['valid'] & ~d['ratio_valid']] = 'unavailable_main_mass_not_above_guard'
    return status


def bootstrap(out, c, hist, observed, weights_path, job, regression=None):
    """Same frozen bootstrap_roi and pairing functions; lean persistence only."""
    width = c['distance_max_um'] / c['distance_bins']
    mid = (np.arange(c['distance_bins']) + .5) * width
    scope_list = mc.scopes(c)
    scalar_rows, paired_rows, valid_rows, control_rows, evidence = [], [], [], [], []
    compute_s = write_s = 0.
    with np.load(weights_path, allow_pickle=False) as saved:
        for length in c['bootstrap']['block_lengths_frames']:
            started = time.perf_counter()
            weights = saved[f'L{length}']
            np.testing.assert_array_equal(weights, mc.shared_resampling_plan(c, length))
            print('BOOTSTRAP_START', length, flush=True)
            groups, controls = [], []
            for g, group in enumerate(mc.GROUPS):
                draws, ctl = mc.bootstrap_roi(hist['total'][g], hist['background'][g], c, length, weights)
                groups.append(draws); controls.append(ctl)
                for (scope, start, stop), d, block_point in zip(scope_list, draws, ctl, strict=True):
                    point = observed[group, scope]
                    control_rows.append(dict(group=group, scope=scope, block_frames=length,
                        possible_block_internal_frame_pairs=stop-start-(stop-start)//length,
                        original_possible_frame_pairs=stop-start-1, **mc.scalar_row(block_point),
                        beta_change_from_full_pairs=block_point['beta'][0]-point['beta'][0],
                        signed_area_change_from_full_pairs=block_point['signed_area'][0]-point['signed_area'][0],
                        max_normalized_shape_change=float(np.max(np.abs(block_point['shape'][0]-point['shape'][0])))/width))
                    for metric in METRICS:
                        lo, hi, n, fraction = mc.finite_ci(d[metric], c['minimum_joint_valid_fraction'], np.isfinite(point[metric][0]))
                        scalar_rows.append(dict(group=group, scope=scope, block_frames=length, metric=metric,
                            point_estimate=point[metric][0], exploratory_low=lo, exploratory_high=hi,
                            valid_replicates=n, valid_fraction=fraction, coverage_validated=False))
            if regression:
                regression(length, groups, controls)
            arrays = dict(replicate_id=np.arange(c['bootstrap']['replicates']), group_order=np.array(mc.GROUPS),
                          scope_order=np.array([s[0] for s in scope_list]))
            for g, group in enumerate(mc.GROUPS):
                for key in groups[g][0]:
                    if key in FIELDS:
                        continue
                    arrays[group+'__'+key] = np.stack([d[key] for d in groups[g]])
                for field in FIELDS:
                    arrays[group+'__'+field+'_valid'] = np.stack([np.isfinite(d[field]).all(1) for d in groups[g]])
                    arrays[group+'__'+field+'_status'] = np.stack([field_status(d, field) for d in groups[g]])
                for metric in METRICS:
                    arrays[group+'__'+metric+'_status'] = np.stack([scalar_status(d, metric) for d in groups[g]])
            for j, (scope, start, stop) in enumerate(scope_list):
                a, b = groups[0][j], groups[1][j]
                pa, pb = observed['On', scope], observed['Not-on', scope]
                columns = dict(distance_mid_um=mid)
                for field in FIELDS:
                    paired = mc.paired_curve_summary(a[field]/width, b[field]/width,
                        pa[field][0]/width, pb[field][0]/width, c['minimum_joint_valid_fraction'])
                    for prefix in ('On', 'Not_on'):
                        for ending in ('low', 'high'):
                            columns[prefix+'_'+field+'_'+ending] = paired[prefix+'_'+ending]
                    columns[field+'_point_difference'] = (pa[field][0]-pb[field][0])/width
                    columns[field+'_difference_low'] = paired['low']
                    columns[field+'_difference_high'] = paired['high']
                    arrays[scope+'__'+field+'__common_valid'] = paired['common']
                    valid_rows.append(dict(scope=scope, block_frames=length, field=field,
                        On_valid_count=int(np.isfinite(a[field]).all(1).sum()),
                        Not_on_valid_count=int(np.isfinite(b[field]).all(1).sum()),
                        common_valid_count=paired['common_count'], common_valid_fraction=paired['common_fraction'],
                        exploratory_range_available=paired['available']))
                before = time.perf_counter()
                tagged(pd.DataFrame(columns).assign(scope=scope, block_frames=length), job).to_csv(
                    out/'bootstrap'/f'paired_bands_{scope}_L{length}.csv', index=False)
                write_s += time.perf_counter()-before
                for metric in METRICS:
                    difference, good, (lo, hi, n, fraction) = mc.paired_difference(a[metric], b[metric],
                        c['minimum_joint_valid_fraction'], np.isfinite(pa[metric][0]) and np.isfinite(pb[metric][0]))
                    arrays[scope+'__'+metric+'__paired_difference'] = difference
                    arrays[scope+'__'+metric+'__common_valid'] = good
                    paired_rows.append(dict(scope=scope, block_frames=length, metric=metric,
                        point_difference=pa[metric][0]-pb[metric][0], exploratory_low=lo, exploratory_high=hi,
                        common_valid_count=n, common_valid_fraction=fraction,
                        paired_by_original_replicate_id=True, coverage_validated=False))
            # Array-byte fingerprints preserve evidence without retaining density banks.
            evidence.append(dict(block_frames=length, density_draws_retained=False,
                shape_density_C_order_SHA256=array_sha(np.stack([np.stack([d['shape']/width for d in draws]) for draws in groups])),
                shape_valid_C_order_SHA256=array_sha(np.stack([np.stack([d['shape_valid'] for d in draws]) for draws in groups]))))
            before = time.perf_counter()
            np.savez_compressed(out/'bootstrap'/f'metric_draws_L{length}.npz', **arrays)
            write_s += time.perf_counter()-before
            compute_s += time.perf_counter()-started
            print('BOOTSTRAP_COMPLETE', length, flush=True)
            del groups, controls, arrays
    for filename, rows in [('bootstrap_scalar_summary.csv', scalar_rows), ('paired_metric_differences.csv', paired_rows),
                           ('bootstrap_curve_validity.csv', valid_rows), ('block_boundary_exclusion_controls.csv', control_rows)]:
        tagged(pd.DataFrame(rows), job).to_csv(out/filename, index=False)
    dump(out/'bootstrap'/'metadata.json', dict(identity=identity(job), weights_path=str(weights_path),
        weights_file_SHA256=sha(weights_path), original_replicate_IDs=list(range(c['bootstrap']['replicates'])),
        group_order=mc.GROUPS, scope_order=[s[0] for s in scope_list], fields=FIELDS,
        density_draws_retained=False, exactly_recomputable_from_raw_counts_and_weights=True,
        unavailable_values='NaN in arrays/CSV; null in JSON; never imputed as zero', paired_by_original_replicate_ID=True,
        formal_inference=False, coverage_validated=False, draw_fingerprints=evidence,
        total_bootstrap_seconds=compute_s, compressed_array_and_band_write_seconds=write_s))


def write_compatibility(out,c,job):
    rule_keys = ['frames','segment_frames','tau_frames','pixel_size_nm','frame_interval_s','distance_max_um','distance_bins',
        'background_fit_um','minimum_background_tail_ordered_pairs','minimum_total_tail_pairs','signed_area_min_abs',
        'main_area_min_positive','main_region_um','tail_region_um','minimum_joint_valid_fraction','group_rule','total_rule',
        'background_rule','normalization_rule','beta_rule','shape_rule','bootstrap','registration_status','calibration_status']
    curve_rules = {k:c[k] for k in rule_keys}
    method_sha = lambda value: hashlib.sha256(json.dumps(jsonable(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    pipeline_rules = dict(curve_definition=curve_rules, fit_rules={k:v for k,v in job.get('fit',{}).items()
        if k not in ['sample_name','sample_id','C1_file','C2_file','roi_file','mat_file','mat_record','crop_xyxy_px',
            'detector_python','fiji','java_home','pilot_segments','tracking','tardis','localization_selection']
        and not k.startswith('expected_')}, classification=dict(k=1,q_tolerance=1e-12,registration_status=c['registration_status']),
        segmentation_rules={k:v for k,v in json.loads((RELEASE/'mask_reference.json').read_text(encoding='utf-8')).items()
            if k in ['source_frame_count','shape_yx','pixel_size_nm','frame_interval_s','support_total_frames','support_minimum_count',
                     'minimum_foreground_area_pixel2','hole_area_exclusive_max_pixel2','per_frame_min32','registration_status']},
        frozen_core_SHA256={p.relative_to(RELEASE/'vendor').as_posix():sha(p) for p in [
            RELEASE/'vendor/mito_tardis/mito_core.py',RELEASE/'vendor/picasso_roi_sensitivity/roi_core.py',
            RELEASE/'vendor/picasso_full_sensitivity/full_core.py',RELEASE/'vendor/nellie_worker.py',
            RELEASE/'vendor/spatial_mask_preview/geometry.py',RELEASE/'vendor/spatial_mask_preview/support50_postmin32_core.py',
            RELEASE/'vendor/real_annular_background/run_real.py',RELEASE/'vendor/real_annular_background/run_full.py',
            RELEASE/'vendor/background_9x9_simulation/calibrated_photometry.py',
            RELEASE/'vendor/background_9x9_simulation/local_background/run_local_background.py',
            RELEASE/'vendor/trackmate_python_bridge/run_detection.py']},
        native_parameters_file_SHA256=sha(RELEASE/'native_parameters.json'))
    dump(out/'compatibility.json', dict(curve_definition=curve_rules,curve_definition_SHA256=method_sha(curve_rules),
        pipeline_method_definition=pipeline_rules,pipeline_method_SHA256=method_sha(pipeline_rules),
        source_ROI_fingerprints_included_in_method_compatibility=False,BR_biological_independence_role=None))


def run(job, hist_override=None, regression=None):
    started = time.perf_counter()
    configure(job); load_science()
    output = Path(job['output']).absolute()
    from batch_common import guard_path, descriptor
    output = guard_path(output)
    canonical = output/'data/localizations.csv.gz'
    before = descriptor(canonical)
    c = json.loads((RELEASE/'tardis_reference.json').read_text(encoding='utf-8'))
    for key in list(c):
        if key.startswith(('expected_', '_')) or key in ['classification_directory', 'mask_directory', 'old_run_for_all_pairs_check',
                'independent_point_reference', 'run_prefix', 'pointer_prefix', 'mask_version', 'output_relative_directory']:
            del c[key]
    c.update(job.get('tardis', {}))
    for key in list(c):
        if key.startswith(('expected_', '_')) or key in ['classification_directory', 'mask_directory', 'old_run_for_all_pairs_check',
                'independent_point_reference', 'run_prefix', 'pointer_prefix', 'mask_version', 'output_relative_directory']:
            del c[key]
    c.update(sample_name=job['sample_name'], sample_id=job['sample_id'], frames=int(job.get('frames',2500)),
             shared_weights=str(RELEASE/'bootstrap_weights.npz'))
    require(c['frames'] == 2500 and c['distance_bins'] == 300 and c['distance_max_um'] == 3, 'Unexpected production rules')
    out = output/'data/tardis'
    if (out/'COMPLETE.json').exists():
        prior = json.loads((out/'metadata.json').read_text(encoding='utf-8'))
        require(prior.get('scientific_fingerprint') == job.get('scientific_fingerprint'),
                'Existing completed analysis belongs to a different scientific fingerprint; use a fresh output')
    for folder in ['', 'distribution/On', 'distribution/Not-on', 'bootstrap', 'pair_histograms']:
        (out/folder).mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(canonical, dtype=str, keep_default_na=False)
    table, keep = selected_table(audit, c)
    masks = output/'masks'
    roi = np.load(masks/'roi_mask.npy', allow_pickle=False)
    vertices = np.load(masks/'roi_vertices_xy.npy', allow_pickle=False)
    require(rc.points_in_polygon(table[['fit_x_px', 'fit_y_px']].to_numpy(), vertices).all(), 'Selected center outside authoritative original ROI')
    frame = pd.DataFrame(dict(frame=np.arange(c['frames']), time_s=np.arange(c['frames'])*c['frame_interval_s']))
    for group in mc.GROUPS+['unknown']:
        frame[group] = table[table.tardis_group.eq(group)].groupby('frame').size().reindex(range(c['frames']), fill_value=0).to_numpy()
    for name, flag in [('QC_excluded', ~keep), ('ROI_candidates', np.ones(len(audit), bool))]:
        frame[name] = audit.loc[flag].assign(frame=lambda d:d.frame.astype(int)).groupby('frame').size().reindex(range(c['frames']), fill_value=0).to_numpy()
    np.testing.assert_array_equal(frame[mc.GROUPS+['unknown', 'QC_excluded']].sum(1), frame.ROI_candidates)
    mapping = np.load(masks/'original_frame_to_mask_block.npy', allow_pickle=False)
    full = np.load(masks/'clean_full_mask.npy', allow_pickle=False)
    analysis = np.load(masks/'analysis_ROI_mask.npy', allow_pickle=False)
    require(full.shape[0] == 50 and full.dtype == bool, 'Unexpected full masks')
    np.testing.assert_array_equal(analysis, full & roi)
    np.testing.assert_array_equal(mapping, np.arange(c['frames'])//50)
    block_full = np.count_nonzero(full, axis=(1, 2)); block_roi = np.count_nonzero(analysis, axis=(1, 2))
    frame['mask_block_index'] = mapping
    frame['mask_full_area_pixel2'] = block_full[mapping]; frame['mask_ROI_area_pixel2'] = block_roi[mapping]
    saved_counts = output/'data/per_frame_counts.csv'
    if saved_counts.exists():
        saved = pd.read_csv(saved_counts)
        for old,new in [('frame','frame'),('On','On'),('Not_on','Not-on'),('unknown','unknown'),
                        ('ROI_candidates','ROI_candidates'),('QC_excluded','QC_excluded')]:
            np.testing.assert_array_equal(saved[old],frame[new])
    tagged(frame, job).to_csv(out/'per_frame_counts.csv', index=False)
    area = frame[['frame', 'time_s']].copy()
    area['block_index'] = mapping
    for scope, values in [('full', block_full[mapping]), ('ROI', block_roi[mapping])]:
        area[scope+'_mask_area_pixel2'] = values
        area[scope+'_mask_area_um2'] = values*(c['pixel_size_nm']/1000)**2
    area['ROI_coverage_percent'] = area.ROI_mask_area_pixel2/int(roi.sum())*100
    tagged(area, job).to_csv(out/'mask_area_by_frame.csv', index=False)
    block_area = area.iloc[np.arange(50)*50].reset_index(drop=True)
    block_area['original_frame_start'] = np.arange(50)*50
    block_area['original_frame_stop_exclusive'] = (np.arange(50)+1)*50
    block_area['source_frames'] = 50
    tagged(block_area, job).to_csv(out/'mask_area_by_block.csv', index=False)
    area_summary = [dict(scope=scope, n_frames=c['frames'], mean_um2=float(area[scope+'_mask_area_um2'].mean()),
                         median_um2=float(area[scope+'_mask_area_um2'].median()), min_um2=float(area[scope+'_mask_area_um2'].min()),
                         max_um2=float(area[scope+'_mask_area_um2'].max())) for scope in ['full', 'ROI']]
    tagged(pd.DataFrame(area_summary), job).to_csv(out/'mask_area_summary.csv', index=False)
    adjacent = []
    for b in range(1, len(full)):
        a, z = analysis[b-1], analysis[b]; union = int((a|z).sum())
        adjacent.append(dict(candidate='support50_postmin32', frame_before=b*50-1, frame_after=b*50,
            block_before=b-1, block_after=b, IoU=float((a&z).sum()/union) if union else 1., scope='ROI'))
    tagged(pd.DataFrame(adjacent), job).to_csv(out/'mask_adjacent_pair_metrics.csv', index=False)
    print('PAIR_HISTOGRAMS_START', len(table), flush=True)
    if hist_override is None:
        hist = mc.calculate_histograms(table, c, lambda n,*_: print('PAIR_FRAME', n, flush=True))
    else:
        with np.load(hist_override, allow_pickle=False) as saved:
            hist = {k: saved[k].copy() for k in ['total', 'background', 'cross', 'same_cross_background', 'all_total', 'all_background', 'background_zero_removed', 'edges_um']}
    check_frames = list(job['sample_frames'])
    require(check_frames == [0,49,200,625,1200,1250,1875,2000,2450,2499], 'Representative-frame contract changed')
    checked = mc.independent_histogram_check(table, hist, c, check_frames)
    write_start = time.perf_counter()
    np.savez_compressed(out/'pair_histograms/per_frame_raw_counts.npz', **hist, group_order=np.array(mc.GROUPS), original_frame=np.arange(c['frames']))
    raw_write = time.perf_counter()-write_start
    observed, summary = {}, []
    for g, group in enumerate(mc.GROUPS):
        th, bh = hist['total'][g], hist['background'][g]
        for scope, start, stop in mc.scopes(c):
            tc, bc = mc.range_histograms(th, bh, start, stop)
            d = mc.decompose_roi(tc, bc, c); observed[group, scope] = d
            curve = mc.distribution_table(tc, bc, d, c).assign(group=group, scope=scope)
            tagged(curve, job).to_csv(out/'distribution'/group/(scope+'.csv'), index=False)
            subset = table[table.tardis_group.eq(group)&table.frame.between(start, stop-1)]
            row = dict(group=group, scope=scope, start_frame=start, stop_exclusive=stop, points=len(subset),
                total_pair_count=int(tc.sum()), background_ordered_nonzero_count=int(bc.sum()),
                background_exact_zero_removed=int(hist['background_zero_removed'][g,start:stop].sum()),
                **mc.scalar_row(d), **mc.tail_diagnostics(tc, bc, th, bh, start, stop, c))
            for key in ['x_se_nm', 'y_se_nm', 'photons_e', 'bg_e_per_pixel', 'sx_px', 'sy_px']:
                for name, q in [('q10', .1), ('median', .5), ('q90', .9)]:
                    row[key+'_'+name] = subset[key].quantile(q)
            summary.append(row)
        pieces = sum(mc.range_histograms(th, bh, s, s+250)[0] for s in range(0,2500,250))
        np.testing.assert_array_equal(pieces+th[np.arange(249,2499,250)].sum(0), th.sum(0))
    tagged(pd.DataFrame(summary), job).to_csv(out/'per_scope_summary.csv', index=False)
    tagged(pd.DataFrame(curve_effects(observed, c)), job).to_csv(out/'descriptive_shape_differences.csv', index=False)
    bootstrap(out, c, hist, observed, RELEASE/'bootstrap_weights.npz', job, regression)
    require(descriptor(canonical) == before, 'Canonical localizations changed during analysis')
    unknown = int(table.tardis_group.eq('unknown').sum())
    write_compatibility(out,c,job)
    dump(out/'metadata.json', dict(identity=identity(job), BR_raw_label=job.get('BR'), BR_biological_independence_role=None,
        original_ROI_authoritative=True, MAT_required=False, canonical_source=str(output/'data/localizations.csv.gz'), canonical_partial_file_fingerprint=before,
        selected_count=len(table), unknown_count=unknown, group_counts=table.tardis_group.value_counts().to_dict(),
        diagnostic_all_pair_name='all_selected' if not unknown else 'known_classified_only',
        config_rules=c, raw_histogram_array_SHA256={k: array_sha(a) for k,a in hist.items()},
        independent_histogram_group_frames_checked=checked, segment_boundary_exclusion_checked=True,
        curve_units='density 1/um; distance um; raw counts ordered pairs as declared',
        nominal_signal_units='manufacturer-nominal photoelectron equivalents', formal_inference=False,
        raw_count_compressed_write_seconds=raw_write, elapsed_seconds=time.perf_counter()-started,
        invalid_reasons=['unavailable_sparse_tail', 'unavailable_invalid_distribution', 'unavailable_near_zero_signed_area',
                         'ratio unavailable when main signed mass is not greater than its positive guard'],
        original_source_ROI_fingerprints_separate_from_curve_rules=True,
        scientific_fingerprint=job.get('scientific_fingerprint'),
        source_fingerprints={'canonical_localizations_partial_descriptor': before, 'ROI_file_SHA256': sha(job['ROI_file']),
                             'roi_mask_C_order_SHA256': array_sha(roi)},
        completed_utc=datetime.now(timezone.utc).isoformat()))
    dump(out/'units_and_invalid_reasons.json', dict(
        units={'frame':'original zero-based camera frame','time_s':'s','fit_x_px':'native C2 pixels','fit_y_px':'native C2 pixels',
               'x_se_nm':'manufacturer-nominal local-model position SE, nm','y_se_nm':'manufacturer-nominal local-model position SE, nm',
               'photons_e':'manufacturer-nominal photoelectron-equivalent whole-PSF signal','bg_e_per_pixel':'nominal electron equivalents/pixel',
               'sx_px':'Gaussian PSF sigma in native pixels','sy_px':'Gaussian PSF sigma in native pixels',
               'distance_um':'um','density':'1/um','raw_pairs':'declared ordered pair counts, not independent samples',
               'beta':'dimensionless background weight, not a molecular state fraction','signed_area':'signed dimensionless mass',
               'mask_area_pixel2':'native binary foreground pixels','mask_area_um2':'native foreground pixels x (0.117 um)^2'},
        unavailable='NaN in CSV/NPZ; null in JSON; never zero-imputed',
        field_status_arrays='bootstrap/metric_draws_L*.npz: group__field_status',
        ratio_rule='main signed mass must be strictly greater than main_area_min_positive; shape validity is separate',
        array_axes={'scalar_draws':['scope','original_replicate_id'],'common_valid':['original_replicate_id']},
        formal_inference=False,coverage_validated=False))
    dump(out/'COMPLETE.json', dict(all_passed=True, numerical_complete=True, visual_review_pending=True,
         selected_count=len(table), unknown_count=unknown, paired_replicates=c['bootstrap']['replicates'], density_banks_retained=False))
    if 'scientific_fingerprint' in job:
        from batch_common import commit_stage
        commit_stage(job, 'analysis', [p for p in sorted(out.rglob('*')) if p.is_file() and not any(part.startswith('.') for part in p.relative_to(out).parts)], numerical_complete=True, visual_review_pending=True)
    print('ANALYSIS_COMPLETE', str(out), flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--job', type=Path)
    parser.add_argument('--windows-reference-replay', '--Windowsreference-replay', dest='reference', type=Path)
    parser.add_argument('--report', type=Path); parser.add_argument('--runtime-dir', type=Path)
    args = parser.parse_args()
    if args.reference:
        if args.job or not args.report:
            parser.error('Reference replay requires --report and cannot be combined with --job')
        from reference_replay import replay
        result=replay(args.reference,args.report,args.runtime_dir)
        print(json.dumps(result,indent=2))
        raise SystemExit(0 if result['all_passed'] else 1)
    if not args.job:
        parser.error('Supply --job or --windows-reference-replay with --report')
    from batch_common import load_job, stage_done
    job = load_job(args.job)
    if stage_done(job, 'analysis'):
        print('ANALYSIS_ALREADY_COMPLETE', flush=True)
    else:
        run(job)


if __name__ == '__main__':
    main()
