"""Read-only adapters for committed legacy cell results, never a refit fallback.

Scientific identities are translated from explicit saved parameter/metadata
contracts. Paths and legacy release/source hashes remain audit provenance only.
"""
from __future__ import annotations
import csv
import gzip
import json
from pathlib import Path
from .types import Artifact
from .identity import scientific_key, normalize, digest
from .cache import file_fingerprint, fingerprint_identity, source_identity
from .contracts import validate_localization_table, validate_mask_contract, PIXEL_CENTERS, ContractError
from .graph import default_specs, stage_parameters

LEGACY_METHODS = {'detection':'trackmate_log_7_14','fit':'annular_integrated_gaussian_observed_hessian','broad':'nellie_broad','support_counts':'nonoverlapping_support_count','mask_threshold':'minimum_support','mask_cleanup':'foreground_and_enclosed_holes','mask_roi':'original_roi_intersection','assignment':'error_ellipse_k1','distances':'tardis_original_adjacent_counts','decomposition':'tardis_signed_decomposition','bootstrap':'paired_stratified_internal_pairs','c1_mean':'raw_adu_arithmetic_mean','statistics':'localization_mask_plot_statistics','presentation':'saved_artifact_presentation'}

class LegacyIncompatible(RuntimeError):
    pass


def _read(path, optional=False):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        if optional:
            return {}
        raise LegacyIncompatible('Missing legacy metadata: ' + str(path))


def _portable(value, config):
    from .config import portable_input
    return portable_input(value, config)


def source_artifacts(config, sample):
    result = {}
    acquisition = config['acquisition']
    for stage, p in [('C1', sample.C1_file), ('C2', sample.C2_file), ('roi', sample.ROI_file)]:
        fp = file_fingerprint(p)
        params = {'source': fingerprint_identity(fp)}
        if stage == 'roi':
            params['raster_grid'] = {'shape_yx': acquisition['shape'][1:], 'pixel_center': acquisition['pixel_center'], 'crop_offset_xy_px': acquisition['crop_offset_xy_px']}
        if stage != 'roi':
            params['acquisition'] = {k: acquisition[k] for k in ('axes', 'shape', 'pixel_size_nm', 'frame_interval_s')}
        key = scientific_key('input_' + stage, '1', params, {})
        result[stage] = Artifact(key, stage, {'roi_file' if stage == 'roi' else 'image': str(Path(p).resolve())},
                                 {'source_fingerprint': fp, 'read_only': True}, params, {}, 'input_' + stage, '1')
    return result


def _csv_rows(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8', newline='') as stream:
        yield from csv.DictReader(stream)


def _require_equal(a, b, message):
    if normalize(a) != normalize(b):
        raise LegacyIncompatible(message)


def _frame_contract(mask, mapping_rows, shape, original_frames):
    rows = sorted(mapping_rows, key=lambda r: int(r['block_index']))
    if [int(r['block_index']) for r in rows] != list(range(shape[0])):
        raise LegacyIncompatible('Legacy block numbering is incomplete')
    blocks = []
    for r in rows:
        if 'original_frame_stop_exclusive' in r: stop = int(r['original_frame_stop_exclusive'])
        elif 'original_frame_end_inclusive' in r: stop = int(r['original_frame_end_inclusive']) + 1
        else: raise LegacyIncompatible('Legacy mapping lacks an explicit inclusive or exclusive endpoint')
        frames = list(range(int(r['original_frame_start']), stop))
        if 'source_frame_count' in r and len(frames) != int(r['source_frame_count']):
            raise LegacyIncompatible('Legacy mapping source frame count contradicts endpoints')
        blocks.append(frames)
    frame_map = {str(f): b for b, fs in enumerate(blocks) for f in fs}
    c = dict(axes='TYX', shape=list(shape), original_frame_ids=list(range(original_frames)),
             frame_to_block=frame_map, block_frame_ids=blocks, expected_block_frame_ids=blocks,
             block_valid=[True] * len(blocks), block_reasons=[''] * len(blocks),
             pixel_size_nm=mask['pixel_size_nm'], frame_interval_s=mask['original_frame_interval_s'],
             crop_offset_xy_px=[0, 0], pixel_center_convention=PIXEL_CENTERS,
             coordinate_transform='identity', registration={'transform': 'identity', 'status': 'unverified'},
             time_semantics='Original frame maps to saved support block; not an instantaneous boundary')
    return validate_mask_contract(c)


class LegacyReader:
    def __init__(self, config):
        self.config = config
        self.root = Path(config['data_root']).resolve()
        self.execution_control = Path(config.get('control_root', self.root / '_batch_pipeline')).resolve()
        # A new batch may own separate state while adopting the same committed
        # legacy results. This explicit location is read-only; never guess roots
        # or copy a completion registry that can silently become stale.
        legacy_control = config.get('legacy_control_relative')
        self.control = (self.root / legacy_control).resolve() if legacy_control else self.execution_control
        if not self.control.is_relative_to(self.root):
            raise ValueError('Legacy metadata root escapes the dataset')
        if legacy_control and not self.control.is_dir():
            raise FileNotFoundError('Explicit legacy metadata root is unavailable: ' + str(self.control))
        self.reports = {}
        self._locations = None
        self._registry_entries = {}

    def _completed_locations(self):
        if self._locations is not None:
            return self._locations
        locations = {}
        def add(sid, output, figures=None, origin=''):
            if not output:
                return
            out = _portable(output, self.config)
            fig = _portable(figures, self.config) if figures else None
            if not out.is_dir():
                return
            candidate = {'output': out, 'figures': fig, 'origin': origin}
            old = locations.get(sid)
            if old and old['output'] != out:
                # Multiple different runs are not chosen by date or row order.
                raise LegacyIncompatible('Conflicting completed runs for ' + sid + ': ' + str(old['output']) + ', ' + str(out))
            locations[sid] = candidate
        for rel in ('windows/state/completed_cells_portable.json', 'state/completed_cells_portable.json'):
            path = self.control / rel
            data = _read(path, True)
            for sid, row in data.get('entries', {}).items():
                self._registry_entries[sid] = row
                if row.get('complete'):
                    add(sid, row.get('result_directory_relative'), row.get('figures_relative'), str(path))
        for rel in ('state/active_batch.json', 'windows/state/active_batch.json'):
            path = self.control / rel
            data = _read(path, True)
            for sid, row in data.get('cells', {}).items():
                if row.get('status') in ('complete', 'completed', 'reused'):
                    add(sid, row.get('output'), row.get('figures'), str(path))
        # Some early Mac cells live solely in the release's completed index.
        release = self.config.get('legacy_release')
        if release:
            for path in (Path(release) / 'completed_index.json', self.control / 'completed_index.json'):
                data = _read(path, True)
                records = data.get('cells', data.get('entries', data)) if isinstance(data, dict) else data
                if isinstance(records, list):
                    records = {r.get('sample_id'): r for r in records}
                if isinstance(records, dict):
                    for sid, row in records.items():
                        if isinstance(row, dict) and row.get('output'):
                            add(sid, row.get('output'), row.get('figures'), str(path))
        self._locations = locations
        return locations

    def discover_completed(self):
        from .config import discover_samples
        result = []
        locations = self._completed_locations()
        for sample in discover_samples(self.config):
            row = locations.get(sample.sample_id)
            if row:
                sample.legacy_output, sample.legacy_figures = row['output'], row['figures']
                result.append(sample)
        return result

    def read_sample(self, sample, validate=True):
        result = source_artifacts(self.config, sample)
        location = self._completed_locations().get(sample.sample_id)
        output = Path(sample.legacy_output) if sample.legacy_output else location['output'] if location else None
        if output is None:
            self.reports[sample.sample_id] = {'compatible': False, 'reason': 'no_committed_legacy_result'}
            return result
        if location:
            sample.legacy_figures = sample.legacy_figures or location['figures']
        sample.legacy_output = output
        job = _read(output / 'job.json')
        if job.get('sample_id') != sample.sample_id:
            raise LegacyIncompatible('Legacy sample identity mismatch')
        self.reports[sample.sample_id] = report = {'compatible': True, 'output': str(output), 'issues': [], 'validation_scope': 'ten representative frames plus table identity/units'}
        index = _read(output / 'RESULT_INDEX.json', True)
        if index.get('reused_existing_analysis') and index.get('source_directories'):
            return self._read_split(sample, result, job, output, index, validate)
        data, masks, render, tardis = output / 'data', output / 'masks', output / 'data/render_sources', output / 'data/tardis'
        mask = _read(masks / 'metadata.json', True)
        definition = _read(self.control / 'scientific_definition.json', True)
        if not definition:
            definition = _read(self.control / 'windows/scientific_definition.json', True)
        defined = definition.get('definition', {})
        if not definition or definition.get('fingerprint') != job.get('scientific_fingerprint'):
            raise LegacyIncompatible('Saved scientific method definition is missing or inconsistent')
        _require_equal(job['frames'], self.config['acquisition']['frames'], 'Legacy original frame count conflicts with acquisition')
        _require_equal(job['pixel_size_nm'], self.config['acquisition']['pixel_size_nm'], 'Legacy pixel units conflict with acquisition')
        _require_equal(job['frame_interval_s'], self.config['acquisition']['frame_interval_s'], 'Legacy time units conflict with acquisition')
        for channel, source in [('C1', sample.C1_file), ('C2', sample.C2_file)]:
            header = job.get('headers', {}).get(channel, {})
            _require_equal(header.get('shape'), self.config['acquisition']['shape'], 'Legacy ' + channel + ' shape conflict')
            _require_equal(header.get('axes'), 'TYX', 'Legacy source axes missing or inconsistent')
        for name, candidates in [('mask', [masks / 'roi_mask.npy', output / 'roi/roi_mask.npy']),
                                 ('vertices', [masks / 'roi_vertices_xy.npy', output / 'roi/vertices.npy'])]:
            matches = [p for p in candidates if p.is_file()]
            if matches:
                result['roi'].files[name] = str(matches[0])
        result['roi'].metadata.update(legacy_output=str(output), read_only=True)
        if validate:
            report['source_checks'] = self._validate_sources(job, output, sample)
        fitpath = data / 'localizations.csv.gz'
        detectionpath = data / 'all_detections.csv.gz'
        if validate and fitpath.is_file() and detectionpath.is_file():
            report['localization'] = validate_localization_table(_csv_rows(fitpath), _csv_rows(detectionpath), frame_interval_s=job['frame_interval_s'])
            summary = _read(data / 'localization_summary.json')
            _require_equal(report['localization']['rows'], summary['ROI_candidates'], 'Legacy localization row count changed')
            _require_equal(report['localization']['selected'], summary['selected'], 'Legacy selected count changed')
        mapping_path = masks / 'block_frame_mapping.csv'
        contract = None
        if mask and mapping_path.is_file():
            contract = _frame_contract(mask, list(_csv_rows(mapping_path)), mask['shape'], job['frames'])
            if validate:
                import numpy as np
                complete = np.load(masks / 'block_completed.npy', allow_pickle=False)
                if complete.dtype != bool or not bool(complete.all()) or len(complete) != mask['shape'][0]:
                    raise LegacyIncompatible('Legacy mask blocks are incomplete')
                report['mask_mapping'] = {'blocks': int(len(complete)), 'original_frames': len(contract['frame_to_block']), 'all_blocks_complete': True}
        legacy_sources = self._legacy_source_ids(job, output, sample, result, masks)
        if legacy_sources['roi'] != result['roi'].id:
            result['roi'].files.pop('mask', None); result['roi'].files.pop('vertices', None)
            result['roi'].metadata['legacy_geometry_not_reused'] = 'original ROI content or raster grid changed'
        params = self._actual_parameters(job, mask, defined, render, tardis)
        specs = {s.stage: s for s in default_specs(self.config)}
        sources = {
            'detection': {'table': detectionpath, 'per_frame_counts': data / 'detection_frame_counts.csv'},
            'fit': {'table': fitpath, 'summary': data / 'localization_summary.json', 'per_frame_counts': data / 'localization_frame_counts.csv'},
            'broad': {'directory': masks / 'broad_packed', 'packed_dir': masks / 'broad_packed', 'metadata': masks / 'metadata.json'},
            'support_counts': {'counts': masks / 'foreground_count.npy', 'mapping': mapping_path, 'frame_map': masks / 'original_frame_to_mask_block.npy'},
            'mask_threshold': {},
            'mask_cleanup': {'full_mask': masks / 'clean_full_mask.npy', 'roi': masks / 'roi_mask.npy', 'vertices': masks / 'roi_vertices_xy.npy', 'mapping': mapping_path, 'frame_map': masks / 'original_frame_to_mask_block.npy', 'counts': masks / 'foreground_count.npy', 'statistics': masks / 'per_block_statistics.csv'},
            'mask_roi': {'roi_mask': masks / 'analysis_ROI_mask.npy', 'statistics': masks / 'per_block_statistics.csv', 'mapping': mapping_path},
            'assignment': {'labels': fitpath, 'summary': output / 'classification_summary.json', 'per_frame_counts': data / 'per_frame_counts.csv'},
            'distances': {'counts': tardis / 'pair_histograms/per_frame_raw_counts.npz', 'metadata': tardis / 'metadata.json'},
            'decomposition': {'directory': tardis, 'metadata': tardis / 'metadata.json', 'summary': tardis / 'per_scope_summary.csv'},
            'bootstrap': {'directory': tardis / 'bootstrap', 'metadata': tardis / 'bootstrap/metadata.json'},
            'c1_mean': {'mean': render / 'C1_mean_frames0000_2499.npy', 'metadata': render / 'C1_average2500_metadata.json'},
            'statistics': {'directory': render, 'metadata': render / 'render_metadata.json', 'histograms': render / 'histogram_counts_and_edges.npz', 'pooled_count_maps': render / 'pooled_count_maps.npz', 'tardis_directory': tardis, 'intra_statistics': render / 'intra_descriptive_statistics.csv'},
            'presentation': {'directory': sample.legacy_figures or _portable(job['figures'], self.config)},
        }
        state_names = {'detection': 'detection', 'fit': 'localization', 'broad': 'mask', 'support_counts': 'mask',
                       'mask_threshold': 'mask', 'mask_cleanup': 'mask', 'mask_roi': 'mask', 'assignment': 'classification',
                       'distances': 'analysis', 'decomposition': 'analysis', 'bootstrap': 'analysis', 'c1_mean': 'render',
                       'statistics': 'render', 'presentation': 'render'}
        for stage, files in sources.items():
            spec = specs[stage]
            if any(dep not in result for dep in spec.dependencies):
                report['issues'].append({'stage': stage, 'reason': 'missing_compatible_legacy_upstream'})
                continue
            state = _read(output / 'state' / (state_names[stage] + '.json'), True)
            if not state.get('complete'):
                report['issues'].append({'stage': stage, 'reason': 'missing_legacy_stage_completion'})
                continue
            if stage not in params:
                report['issues'].append({'stage': stage, 'reason': 'insufficient_legacy_method_parameters'})
                continue
            missing = [str(p) for p in files.values() if not p.exists()]
            if missing:
                report['issues'].append({'stage': stage, 'reason': 'missing_legacy_files', 'files': missing})
                continue
            if validate:
                self._validate_files(files, state, output)
            # Preserve actual old parameters, not the currently requested values.
            actual = params[stage]
            if set(actual) != set(stage_parameters(self.config, spec)):
                report['issues'].append({'stage': stage, 'reason': 'new_or_unrecognized_parameter_fields'})
                continue
            upstream = {d: legacy_sources[d] if d in legacy_sources else result[d].id for d in spec.dependencies}
            legacy_method = LEGACY_METHODS[stage]
            key = scientific_key(legacy_method, '1', actual, upstream)
            metadata = {'legacy_output': str(output), 'read_only': True, 'legacy_scientific_fingerprint': job['scientific_fingerprint'], 'legacy_adapter': 'explicit-v1'}
            metadata.update(pixel_size_nm=job['pixel_size_nm'], frame_interval_s=job['frame_interval_s'])
            if contract and stage in ('broad', 'support_counts', 'mask_threshold', 'mask_cleanup', 'mask_roi', 'assignment'):
                metadata['contract'] = contract
            if stage == 'mask_threshold':
                metadata.update(virtual=True, materialization_required=True, explanation='Threshold can be recovered from saved support counts; no intermediate array was retained')
            if stage == 'assignment':
                metadata.update(uid_field='source_detection_uid', label_field='postmin32_pointwise_label',
                                selected_field='spatial_selected', counts=_read(output / 'classification_summary.json'))
            if stage == 'statistics':
                metrics = self._saved_scalar_metrics(sample)
                if metrics is not None: metadata['metrics'] = metrics
            if stage == 'fit':
                metadata.update(uid_field='source_detection_uid', selected_field='spatial_selected', original_fields_preserved=True)
            result[stage] = Artifact(key, stage, {k: str(p) for k, p in files.items()}, metadata, actual, upstream, legacy_method, '1')
        report['compatible_stages'] = list(result)
        return result

    def _actual_parameters(self, job, mask, definition, render, tardis, overrides=None):
        """Translate only documented fields from the known legacy method contract."""
        import numpy as np
        common = {'frames': job['frames'], 'shape': job['headers']['C1']['shape'][1:],
                  'pixel_size_nm': job['pixel_size_nm'], 'frame_interval_s': job['frame_interval_s']}
        oldfit = job.get('fit', {})
        overrides = overrides or {}
        rules = overrides.get('rules', _read(tardis / 'metadata.json', True).get('config_rules', {}))
        native = definition.get('native_parameters', {})
        # The caller's output can be rebased from Windows; classification k is
        # also explicitly present in the immutable ellipse method definition.
        if definition.get('ellipse') != 'k1 diagonal SE closed pixel squares tangent included tolerance1e-12':
            raise LegacyIncompatible('Unsupported or unspecified legacy ellipse semantics')
        result = {}
        fit_keys = ('box_px', 'annulus_inner_radius_px', 'annulus_outer_radius_px', 'neighbor_core_radius_px',
                    'minimum_background_pixels', 'minimum_jackknife_sectors', 'width_bounds_px',
                    'initial_sigma_multistart_px', 'fit_max_iterations', 'calibration_status', 'noise_model', 'camera')
        detector_keys = ('trackmate_version', 'radius_um', 'threshold', 'median_filter', 'subpixel_localization')
        if all(k in oldfit for k in detector_keys):
            result['detection'] = dict(common, **{k: oldfit[k] for k in detector_keys})
        if all(k in oldfit for k in fit_keys):
            result['fit'] = dict(common, **{k: oldfit[k] for k in fit_keys})
        if all(k in native for k in ('filter_min_radius_um', 'filter_max_radius_um', 'version')):
            result['broad'] = dict(common, filter_radii_um=[native['filter_min_radius_um'], native['filter_max_radius_um']], nellie_version=native['version'])
        if mask and definition.get('mask') == 'native broad -> support>=10/50 -> FG8>=32 -> enclosed BG4<64 -> ROI':
            result['support_counts'] = {'window_frames': mask['mask_block_frames'], 'stride_frames': mask['mask_block_frames'],
                                       'frame_ids': list(range(mask['original_frames'])), 'count_rule': 'sum_bool_per_pixel', 'partial_window_policy': 'unknown'}
            result['mask_threshold'] = {'minimum_support': mask['foreground_support_minimum']}
            result['mask_cleanup'] = {'min_foreground_pixels': mask['minimum_foreground_area_pixel2'],
                                      'foreground_connectivity': mask['foreground_connectivity'],
                                      'max_hole_area_exclusive': mask['hole_area_exclusive_max_pixel2'],
                                      'background_connectivity': mask['hole_background_connectivity']}
            result['mask_roi'] = {}
        result['assignment'] = dict(common, k=1.0, tolerance=1e-12, closed_pixel_squares=True,
                                   covariance='diagonal_axis_aligned', registration={'transform': 'identity', 'status': 'unverified'}, unknown_priority=True)
        counts = tardis / 'pair_histograms/per_frame_raw_counts.npz'
        if counts.is_file():
            with np.load(counts, allow_pickle=False) as bank:
                edges = bank['edges_um'].tolist()
            result['distances'] = {k: rules[k] for k in ('frames', 'pixel_size_nm', 'frame_interval_s', 'tau_frames', 'total_rule', 'background_rule') if k in rules}
            result['distances']['edges_um'] = edges
        decomp_keys = ('frames', 'segment_frames', 'tau_frames', 'pixel_size_nm', 'frame_interval_s', 'distance_max_um',
                       'distance_bins', 'background_fit_um', 'minimum_background_tail_ordered_pairs', 'minimum_total_tail_pairs',
                       'signed_area_min_abs', 'main_area_min_positive', 'main_region_um', 'tail_region_um',
                       'minimum_joint_valid_fraction', 'normalization_rule', 'beta_rule', 'shape_rule')
        if all(k in rules for k in decomp_keys):
            result['decomposition'] = {k: rules[k] for k in decomp_keys}
        if 'bootstrap' in rules:
            result['bootstrap'] = dict(rules['bootstrap'])
        meanmeta = overrides.get('mean', _read(render / 'C1_average2500_metadata.json', True))
        if meanmeta and all(k in meanmeta for k in ('sum_uint64_C_bytes_SHA256', 'mean_float32_C_bytes_SHA256', 'source_frames', 'all_source_pixels_included', 'filtering')):
            result['c1_mean'] = dict(common, include_zeros=bool(meanmeta['all_source_pixels_included'] and not meanmeta.get('zero_exclusion', True)),
                                     accumulator='uint64', output_dtype='float32', temporal_filtering=meanmeta['filtering'])
        if 'mean_parameters' in overrides:
            result['c1_mean'] = dict(common, **overrides['mean_parameters'])
        histogram = overrides.get('histogram', _read(render / 'histogram_metadata.json', True))
        pooled = overrides.get('pooled', _read(render / 'pooled_metadata.json', True))
        if histogram and pooled and 'SE_edges_nm' in histogram and 'photon_edges_e' in histogram and 'shape_rule' in rules:
            result['statistics'] = dict(common, SE_edges_nm=histogram['SE_edges_nm'], signal_bins=len(histogram['photon_edges_e']) - 1,
                                         signal_axis='linear', selection='original_QC_and_refined_center_ROI',
                                         on_percentage_denominator='On+Not_on', mask_mean='time_weighted_blocks',
                                         pooling='all_original_frames_repeated_observations_retained', shape_rule=rules['shape_rule'])
        elif histogram and pooled and 'SE_edges_nm' in histogram and 'signal_edges_e' in histogram and 'shape_rule' in rules:
            result['statistics'] = dict(common, SE_edges_nm=histogram['SE_edges_nm'], signal_bins=len(histogram['signal_edges_e']) - 1,
                                         signal_axis='linear', selection='original_QC_and_refined_center_ROI',
                                         on_percentage_denominator='On+Not_on', mask_mean='time_weighted_blocks',
                                         pooling='all_original_frames_repeated_observations_retained', shape_rule=rules['shape_rule'])
        # Presentation identity is separate. These are the explicit old display
        # settings of the registered legacy renderer, not scientific parameters.
        result['presentation'] = {'style': {'C1_display_ADU': oldfit.get('display_C1_ADU'),
                                          'C2_display_ADU': oldfit.get('display_C2_ADU'),
                                          'on_color': '#00CFA2', 'not_on_color': '#B885E3', 'unknown_color': '#A0A0A0',
                                          'pooled_scale': pooled.get('display_scale'), 'dot_radius_px': pooled.get('dot_radius_native_px'),
                                          'representative_scale': 2}}
        return result


    def _validate_sources(self, job, output, sample):
        import hashlib
        import numpy as np
        import tifffile
        from .contracts import TEN_FRAMES
        saved = _read(output / 'sampled_input_fingerprints.json')
        checked = {}
        for key, path in [('C1_file', sample.C1_file), ('C2_file', sample.C2_file)]:
            expected = {int(r['frame']): r.get('pixel_sha256', r.get('pixel_SHA256'))
                        for r in saved.get('sampled_pixels', {}).get(key, [])}
            if not set(TEN_FRAMES).issubset(expected):
                raise LegacyIncompatible('Legacy source lacks the ten declared sample-frame pixel fingerprints: ' + key)
            with tifffile.TiffFile(path) as tf:
                if len(tf.pages) != job['frames']:
                    raise LegacyIncompatible('Legacy source frame count changed: ' + key)
                for frame in TEN_FRAMES:
                    image = tf.pages[frame].asarray()
                    if list(image.shape) != job['headers'][key[:2]]['shape'][1:] or image.dtype != np.uint16:
                        raise LegacyIncompatible('Legacy source shape/dtype changed')
                    digest_value = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
                    if digest_value != expected[frame]:
                        raise LegacyIncompatible('Legacy source sample pixels changed: ' + key + ' frame ' + str(frame))
            checked[key] = list(TEN_FRAMES)
        roi = saved.get('source_identity', {}).get('ROI', {})
        if roi.get('sha256') != file_fingerprint(sample.ROI_file)['digest']:
            raise LegacyIncompatible('Original ROI content fingerprint is missing or changed')
        return checked


    def _validate_files(self, files, state, output):
        import hashlib
        # ROI fits were intentionally removed by old cleanup. The preserved master
        # table is validated using its classification commit instead.
        records = list(state.get('artifacts', []))
        records += _read(output / 'state/classification.json', True).get('artifacts', [])
        by_path = {}
        for record in records:
            try:
                path = _portable(record.get('path', ''), self.config)
            except ValueError:
                continue
            by_path[str(path)] = record
        for name, path in files.items():
            if path.is_dir():
                continue
            record = by_path.get(str(path))
            if not record:
                # Nested CSV/NPZ products are committed as part of analysis/render;
                # each consumed file must still exist, and its small metadata is
                # inspected by the parameter adapter. No fabricated old hash.
                continue
            if path.stat().st_size != record.get('bytes'):
                raise LegacyIncompatible('Legacy committed file size changed: ' + str(path))
            expected = record.get('sha256') or record.get('sampled_sha256')
            if not expected:
                raise LegacyIncompatible('Legacy file has no content fingerprint: ' + str(path))
            h = hashlib.sha256()
            with path.open('rb') as f:
                if record.get('sha256'):
                    for chunk in iter(lambda: f.read(1024 * 1024), b''):
                        h.update(chunk)
                else:
                    h.update(f.read(65536)); f.seek(max(0, path.stat().st_size - 65536)); h.update(f.read(65536))
            if h.hexdigest() != expected:
                raise LegacyIncompatible('Legacy committed file content changed: ' + str(path))


    def _legacy_source_ids(self, job, output, sample, sources, masks=None):
        """Bind old products to old source fingerprints, not current source bytes."""
        import hashlib
        import numpy as np
        import tifffile
        saved = _read(output / 'sampled_input_fingerprints.json')
        binding = _read(Path(masks) / 'source_binding.json', True) if masks else {}
        self._completed_locations()
        registry = self._registry_entries.get(sample.sample_id, {})
        ids = {}
        for stage in ('C1', 'C2', 'roi'):
            name = 'ROI' if stage == 'roi' else stage
            key = name + '_file'
            raw = registry.get('source_files', {}).get(key, {}) or binding.get(name, {}) or saved.get('source_identity', {}).get(name, {})
            checksum = raw.get('sha256') or raw.get('sampled_sha256')
            if checksum:
                fp = {'bytes': raw['bytes'], 'digest': checksum,
                      'kind': 'complete_sha256' if raw.get('sha256') else 'first_last_64KiB'}
                params = {'source': fp}
                if stage == 'roi':
                    params['raster_grid'] = {'shape_yx': job['headers']['C1']['shape'][1:], 'pixel_center': 'integer_xy_centers', 'crop_offset_xy_px': [0, 0]}
                if stage != 'roi':
                    params['acquisition'] = {'axes': job['headers'][stage]['axes'], 'shape': job['headers'][stage]['shape'],
                                             'pixel_size_nm': job['pixel_size_nm'], 'frame_interval_s': job['frame_interval_s']}
                ids[stage] = scientific_key('input_' + stage, '1', params, {})
            elif stage in ('C1', 'C2'):
                # Early Mac sources recorded ten page hashes rather than byte
                # samples. Use those declared source checks; never infer from mtime.
                reference = saved.get('sampled_pixels', {}).get(stage + '_file', [])
                if not reference:
                    raise LegacyIncompatible('No saved source content evidence for ' + stage)
                actual = []
                with tifffile.TiffFile(getattr(sample, stage + '_file')) as tf:
                    for row in reference:
                        frame = int(row['frame']); h = hashlib.sha256(np.ascontiguousarray(tf.pages[frame].asarray()).tobytes()).hexdigest()
                        actual.append(h == row.get('pixel_sha256', row.get('pixel_SHA256')))
                ids[stage] = sources[stage].id if all(actual) else digest({'legacy_source_pages': reference, 'channel': stage})
            else:
                raise LegacyIncompatible('No saved ROI content fingerprint')
        return ids


    def _read_split(self, sample, result, job, output, index, validate):
        """The original verified T1 stores explicit source-directory references."""
        import numpy as np
        d = {k: _portable(v, self.config) for k, v in index['source_directories'].items()}
        required = ('localization_cache','mask_directory','classification_directory','TARDIS_directory','C1_average_directory','supplemental_directory')
        if any(k not in d or not d[k].is_dir() for k in required):
            raise LegacyIncompatible('Verified split legacy source directory is unavailable')
        complete = _read(output / 'COMPLETED.json')
        if not complete.get('complete') or complete.get('sample_id') != sample.sample_id:
            raise LegacyIncompatible('Verified split wrapper is not committed')
        report = {'compatible':True,'output':str(output),'legacy_layout':'explicit_split_source_directories','issues':[],
                  'validation_scope':'ten representative source frames plus persisted UID and error-field contract'}
        self.reports[sample.sample_id] = report
        fit, masks, classification, tardis, average, supplemental = (d[k] for k in required)
        maskparams = _read(masks / 'parameters.json')
        mask = dict(shape=maskparams['shape'],axes=maskparams['axes'],original_frames=maskparams['source_frame_count'],
                    original_frame_interval_s=maskparams['frame_interval_s'],mask_block_frames=maskparams['support_total_frames'],
                    pixel_size_nm=maskparams['pixel_size_nm'],foreground_support_minimum=maskparams['support_minimum_count'],
                    foreground_connectivity=maskparams['foreground_connectivity'],minimum_foreground_area_pixel2=maskparams['minimum_foreground_area_pixel2'],
                    hole_background_connectivity=maskparams['background_connectivity'],hole_area_exclusive_max_pixel2=maskparams['hole_area_exclusive_max_pixel2'])
        contract = _frame_contract(mask,list(_csv_rows(masks/'block_frame_mapping.csv')),mask['shape'],job['frames'])
        job = dict(job,fit=_read(fit/'config.json'))
        definition = _read(self.control/'scientific_definition.json')
        if definition.get('fingerprint') != job['scientific_fingerprint']:
            raise LegacyIncompatible('Verified T1 scientific definition conflict')
        rules = _read(tardis/'config.json')
        hist = _read(supplemental/'precision_signal_histograms/histogram_metadata.json')
        if 'SE_bins_nm' in hist: hist['SE_edges_nm']=hist['SE_bins_nm']
        if 'photons_bins_e' in hist: hist['photon_edges_e']=hist['photons_bins_e']
        meanparams = _read(average/'parameters.json')
        if meanparams.get('method') != 'pixelwise arithmetic time mean, float64 accumulator dividedby2500; scientificfloat32':
            raise LegacyIncompatible('Verified mean arithmetic semantics are unavailable')
        bound = job['frames'] * 65535
        if bound >= 2**53 or not meanparams.get('zeros_included') or meanparams.get('filtering'):
            raise LegacyIncompatible('Verified mean cannot be mapped to exact integer-sum identity')
        overrides = {'rules':rules,'histogram':hist,
                     'pooled':{'display_scale':meanparams['display_scale'],'dot_radius_native_px':meanparams['dot_radius_native_px']},
                     'mean_parameters':{'include_zeros':True,'accumulator':'uint64','output_dtype':'float32','temporal_filtering':False}}
        params = self._actual_parameters(job,mask,definition['definition'],supplemental/'precision_signal_histograms',tardis,overrides)
        sources = self._legacy_source_ids(job,output,sample,result,None)
        if sources['roi'] == result['roi'].id:
            result['roi'].files.update(mask=str(masks/'roi_mask.npy'),vertices=str(masks/'roi_vertices_xy.npy'))
        else:
            result['roi'].metadata['legacy_geometry_not_reused'] = 'original ROI content or raster grid changed'
        master = classification/'classification/all_ROI_candidates_postmin32.csv'
        detect = fit/'all_detections.csv'
        if validate:
            report['source_checks'] = self._validate_sources(job,output,sample)
            # Original T1 fit records intentionally do not duplicate detector X/Y.
            # Join persisted UID/frame; absence is reported rather than reconstructed.
            report['localization'] = validate_localization_table(_csv_rows(master),_csv_rows(detect),frame_interval_s=job['frame_interval_s'])
            report['localization']['original_detector_coordinates_duplicated_in_fit_table'] = False
            report['mask_blocks_complete'] = bool(np.load(masks/'block_completed.npy',allow_pickle=False).all())
            if not report['mask_blocks_complete']:
                raise LegacyIncompatible('Verified split mask block is incomplete')
        files = {
            'detection':{'table':detect,'per_frame_counts':fit/'frame_counts.csv'},
            'fit':{'table':master,'summary':fit/'summary.json','per_frame_counts':fit/'frame_counts.csv'},
            'broad':{'array':masks/'original_broad_masks.npy','frame_ids':masks/'original_frame_numbers.npy','metadata':masks/'parameters.json'},
            'support_counts':{'counts':masks/'foreground_count.npy','mapping':masks/'block_frame_mapping.csv','frame_map':masks/'original_frame_to_mask_block.npy'},
            'mask_threshold':{'mask':masks/'input_ge10_full_mask.npy'},
            'mask_cleanup':{'full_mask':masks/'clean_full_mask.npy','roi':masks/'roi_mask.npy','vertices':masks/'roi_vertices_xy.npy','mapping':masks/'block_frame_mapping.csv','frame_map':masks/'original_frame_to_mask_block.npy','statistics':masks/'per_block_statistics.csv'},
            'mask_roi':{'roi_mask':masks/'analysis_ROI_mask.npy','statistics':masks/'per_block_statistics.csv','mapping':masks/'block_frame_mapping.csv'},
            'assignment':{'labels':master,'summary':classification/'classification/summary.json','per_frame_counts':classification/'classification/per_frame_counts.csv'},
            'distances':{'counts':tardis/'pair_histograms/per_frame_raw_counts.npz','metadata':tardis/'config.json'},
            'decomposition':{'directory':tardis,'metadata':tardis/'config.json','summary':tardis/'per_scope_summary.csv'},
            'bootstrap':{'directory':tardis/'bootstrap','metadata':tardis/'config.json'},
            'c1_mean':{'mean':average/'C1_mean_frames0000_2499.npy','metadata':average/'parameters.json'},
            'statistics':{'directory':supplemental,'metadata':supplemental/'metadata.json','histograms':supplemental/'precision_signal_histograms/histogram_counts_and_edges.npz','pooled_count_maps':classification/'projections/localization_count_maps.npz','tardis_directory':tardis,'intra_statistics':tardis/'intra_descriptive_statistics.csv'},
            'presentation':{'directory':sample.legacy_figures or _portable(job['figures'],self.config)}
        }
        for spec in default_specs(self.config):
            stage = spec.stage
            if any(dep not in result for dep in spec.dependencies):
                report['issues'].append({'stage':stage,'reason':'missing_compatible_legacy_upstream'});continue
            if stage not in params or set(params[stage])!=set(stage_parameters(self.config,spec)):
                report['issues'].append({'stage':stage,'reason':'insufficient_legacy_parameter_fields'});continue
            missing = [str(p)for p in files[stage].values()if not p.exists()]
            if missing:
                report['issues'].append({'stage':stage,'reason':'missing_legacy_files','files':missing});continue
            upstream = {dep:sources[dep]if dep in sources else result[dep].id for dep in spec.dependencies}
            method = LEGACY_METHODS[stage]
            key = scientific_key(method,'1',params[stage],upstream)
            metadata = {'legacy_output':str(output),'read_only':True,'legacy_split':True,'config_rules':rules,
                        'pixel_size_nm':job['pixel_size_nm'],'frame_interval_s':job['frame_interval_s']}
            if stage in ('broad','support_counts','mask_threshold','mask_cleanup','mask_roi','assignment'):
                metadata['contract'] = contract
            if stage=='fit':metadata.update(uid_field='source_detection_uid',selected_field='qc_unflagged_model_fit',original_fields_preserved=True)
            if stage=='assignment':metadata.update(uid_field='source_detection_uid',label_field='postmin32_pointwise_label',selected_field='qc_unflagged_model_fit',counts=index['counts'])
            if stage=='statistics':
                metrics = self._saved_scalar_metrics(sample)
                if metrics is not None:metadata['metrics']=metrics
            if stage=='c1_mean':metadata['integer_sum_equivalence']={'legacy_accumulator':'float64','canonical_accumulator':'uint64','maximum_sum':bound,'float64_integer_exact_limit':2**53,'source_dtype':'uint16','same_float32_scientific_mean':True}
            result[stage] = Artifact(key,stage,{k:str(p)for k,p in files[stage].items()},metadata,params[stage],upstream,method,'1')
        report['compatible_stages']=list(result)
        return result


    def _saved_scalar_metrics(self, sample):
        status = _read(self.control / 'cell_metrics_status.json', True)
        nodes = [r for r in status.get('nodes', []) if r.get('key') == sample.sample_id and r.get('level') == 'cell']
        if not nodes:
            return None
        if len(nodes) != 1:
            raise LegacyIncompatible('Conflicting saved scalar metric nodes for ' + sample.sample_id)
        directory = _portable(nodes[0]['data'], self.config)
        rows = list(_csv_rows(directory / 'cell_metrics.csv'))
        if len(rows) != 1 or rows[0].get('sample_id') != sample.sample_id:
            raise LegacyIncompatible('Saved scalar metrics do not identify exactly one cell')
        row = rows[0]
        keys = ('On','Not_on','unknown','QC_excluded','selected','ROI_candidates')
        required = (*keys,'on_percentage','first_mask_ROI_area_um2','mean_ROI_area_um2')
        if any(row.get(k) in ('',None) for k in required):
            raise LegacyIncompatible('Saved scalar metric fields are unavailable')
        result = {k:int(row[k]) for k in keys}
        result.update({k:float(row[k])for k in required if k not in keys})
        if result['On']+result['Not_on']+result['unknown']!=result['selected'] or result['selected']+result['QC_excluded']!=result['ROI_candidates']:
            raise LegacyIncompatible('Saved scalar metric count conservation failed')
        result['source_table'] = str(directory/'cell_metrics.csv')
        return result



def discover_completed(config):
    return LegacyReader(config).discover_completed()













