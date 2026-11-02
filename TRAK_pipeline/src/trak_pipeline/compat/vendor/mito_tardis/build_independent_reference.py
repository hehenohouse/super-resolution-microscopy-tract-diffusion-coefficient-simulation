#!/usr/bin/env python3
"""Independently reconstruct tau1 raw histograms from completed classifications.

Uses only Python CSV parsing and NumPy broadcast dx/dy/sqrt + histogram. No
production histogram, distance kernel, decomposition or point estimate is called.
The only write is the requested fresh JSON inside this project workspace.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

VERSION = '1.0.0'
PROJECT = Path(__file__).resolve().parent.parent
GROUPS = ('On', 'Not-on')
LABELS = {'mito-compatible': 'On', 'non-mito-compatible': 'Not-on', 'unknown': 'unknown'}
RULES = dict(frames=2500, tau_frames=1, pixel_size_nm=117.0, frame_interval_s=0.02,
    distance_max_um=3.0, distance_bins=300, background_fit_um=[2.1, 3.0],
    minimum_background_tail_ordered_pairs=100, minimum_total_tail_pairs=100,
    signed_area_min_abs=0.01, main_area_min_positive=0.01,
    main_region_um=[0.0, 0.1], tail_region_um=[0.1, 0.25])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=file_sha(path))


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order='C')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_selected(path, label_field, frames, pixel_size_nm, frame_interval_s):
    """Python float parses each CSV number with round-trip binary64 precision."""
    coordinates = {group: [[] for _ in range(frames)] for group in (*GROUPS, 'unknown')}
    selected_count = 0
    candidates = set()
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'frame', 'fit_x_px', 'fit_y_px', label_field}
        require(required <= set(reader.fieldnames or ()), f'Selected CSV missing fields: {required}')
        for row in reader:
            frame = int(row['frame'])
            require(0 <= frame < frames, f'Original frame outside [0,{frames - 1}]: {frame}')
            label = row[label_field]
            require(label in LABELS, f'Unrecognized frozen classification label: {label!r}')
            xy = (float(row['fit_x_px']), float(row['fit_y_px']))
            require(bool(np.isfinite(xy).all()), 'Selected coordinates must be finite')
            for key in ('qc_unflagged_model_fit', 'fitted_center_inside_ROI', 'spatial_selected'):
                if key in row:
                    require(row[key].strip().lower() in ('true', '1'), f'Unselected row in selected CSV: {key}')
            if 'time_s' in row:
                require(abs(float(row['time_s']) - frame * frame_interval_s) <= 1e-12,
                        f'Time does not match original frame {frame}')
            if 'candidate_index' in row:
                require(row['candidate_index'] not in candidates, 'Duplicate candidate_index in selected CSV')
                candidates.add(row['candidate_index'])
            coordinates[LABELS[label]][frame].append(xy)
            selected_count += 1
    # The order of arithmetic is deliberate: match native xy*117/1000, rather
    # than rounding the combined scale to .117 before the multiplication.
    coordinates = {group: [np.asarray(rows, dtype=np.float64).reshape(-1, 2) * pixel_size_nm / 1000
                           for rows in by_frame] for group, by_frame in coordinates.items()}
    return coordinates, selected_count


def distance_histogram(left, right, edges, *, remove_zero=False, max_broadcast_pairs=2_000_000):
    """Count every Cartesian pair; same-frame zeros include coincident points."""
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    zero_removed = 0
    if not len(left) or not len(right):
        return counts, zero_removed
    chunk_rows = max(1, max_broadcast_pairs // len(right))
    for start in range(0, len(left), chunk_rows):
        piece = left[start:start + chunk_rows]
        dx = piece[:, 0, None] - right[None, :, 0]
        dy = piece[:, 1, None] - right[None, :, 1]
        distances = np.sqrt(dx * dx + dy * dy).ravel()
        if remove_zero:
            zero_removed += int(np.count_nonzero(distances == 0))
            distances = distances[distances > 0]
        counts += np.histogram(distances, bins=edges)[0].astype(np.int64)
    return counts, zero_removed


def reconstruct(coordinates, *, frames=2500, bins=300, distance_max_um=3.0):
    """Keep original time indices, including empty frames and final zero Total."""
    require(all(len(coordinates[group]) == frames for group in GROUPS), 'Frame count mismatch')
    edges = np.linspace(0, distance_max_um, bins + 1, dtype=np.float64)
    arrays = {name: np.zeros((2, frames, bins), dtype=np.int64)
              for name in ('total', 'background', 'cross')}
    arrays.update({name: np.zeros((frames, bins), dtype=np.int64)
                   for name in ('same_cross_background', 'all_total', 'all_background')})
    arrays['background_zero_removed'] = np.zeros((2, frames), dtype=np.int64)
    arrays['edges_um'] = edges
    for frame in range(frames):
        now = [coordinates[group][frame] for group in GROUPS]
        known_now = np.concatenate(now)
        for group in range(2):
            counts, zeros = distance_histogram(now[group], now[group], edges, remove_zero=True)
            arrays['background'][group, frame] = counts
            arrays['background_zero_removed'][group, frame] = zeros
        # Calculate both ordered directions explicitly; do not borrow the
        # production implementation's factor-of-two shortcut.
        arrays['same_cross_background'][frame] = (
            distance_histogram(now[0], now[1], edges, remove_zero=True)[0]
            + distance_histogram(now[1], now[0], edges, remove_zero=True)[0])
        arrays['all_background'][frame] = distance_histogram(known_now, known_now, edges, remove_zero=True)[0]
        if frame + 1 < frames:
            following = [coordinates[group][frame + 1] for group in GROUPS]
            for group in range(2):
                arrays['total'][group, frame] = distance_histogram(now[group], following[group], edges)[0]
                arrays['cross'][group, frame] = distance_histogram(now[group], following[1 - group], edges)[0]
            arrays['all_total'][frame] = distance_histogram(known_now, np.concatenate(following), edges)[0]
    require(np.array_equal(arrays['total'].sum(axis=0) + arrays['cross'].sum(axis=0), arrays['all_total']),
            'Within/cross Total decomposition failed')
    require(np.array_equal(arrays['background'].sum(axis=0) + arrays['same_cross_background'],
                           arrays['all_background']), 'Within/cross background decomposition failed')
    require(not arrays['total'][:, -1].any() and not arrays['cross'][:, -1].any()
            and not arrays['all_total'][-1].any(), 'Final source frame must have no adjacent-frame pairs')
    return arrays


def support_summaries(arrays, rules):
    midpoint = (arrays['edges_um'][:-1] + arrays['edges_um'][1:]) / 2
    tail = (midpoint >= rules['background_fit_um'][0]) & (midpoint < rules['background_fit_um'][1])
    summaries = []
    for group, name in enumerate(GROUPS):
        total = arrays['total'][group].sum(axis=0)
        background = arrays['background'][group].sum(axis=0)
        nt, nb = int(total[tail].sum()), int(background[tail].sum())
        summaries.append(dict(group=name, total_pair_count=int(total.sum()),
            background_ordered_nonzero_pair_count=int(background.sum()),
            total_nonempty_bins=int(np.count_nonzero(total)), background_nonempty_bins=int(np.count_nonzero(background)),
            total_tail_count=nt, background_tail_count=nb,
            total_tail_nonempty_bins=int(np.count_nonzero(total[tail])),
            background_tail_nonempty_bins=int(np.count_nonzero(background[tail])),
            background_exact_zero_removed=int(arrays['background_zero_removed'][group].sum()),
            tail_supported=bool(nt >= rules['minimum_total_tail_pairs']
                                and nb >= rules['minimum_background_tail_ordered_pairs'])))
    return summaries


def build_reference(classification_dir):
    started = time.perf_counter()
    root = Path(classification_dir).resolve()
    config_path, completed_path = root / 'resolved_config.json', root / 'COMPLETED.json'
    csv_path = root / 'classification/selected_localizations_postmin32.csv'
    summary_path = root / 'classification/summary.json'
    protected_paths = [config_path, completed_path, csv_path, summary_path]
    before = [fingerprint(path) for path in protected_paths]
    config, completed, summary = read_json(config_path), read_json(completed_path), read_json(summary_path)
    require(completed.get('delivery_checks_passed') is True, 'Classification delivery is not completed/passed')
    require(completed.get('visual_review_pending', False) is False, 'Classification visual review is pending')
    rules = dict(RULES)
    frame_counts = [config[key] for key in ('frame_count', 'source_frame_count', 'frames') if key in config]
    require(frame_counts and all(value == rules['frames'] for value in frame_counts), 'Expected all 2500 original frames')
    require(config['pixel_size_nm'] == rules['pixel_size_nm']
            and config['frame_interval_s'] == rules['frame_interval_s'], 'Unexpected spatial/time calibration')
    label_field = config.get('classification_label_field', 'postmin32_pointwise_label')
    require(label_field == 'postmin32_pointwise_label', 'Expected frozen postmin32 labels')
    coordinates, selected_count = read_selected(csv_path, label_field, rules['frames'],
                                                rules['pixel_size_nm'], rules['frame_interval_s'])
    group_counts = {name: sum(len(rows) for rows in coordinates[name]) for name in (*GROUPS, 'unknown')}
    require(selected_count == sum(group_counts.values()) == summary['selected'], 'Selected/group counts mismatch')
    for name, field in [('On', 'On'), ('Not-on', 'Not_on'), ('unknown', 'unknown')]:
        require(group_counts[name] == summary[field], f'Classification summary mismatch: {name}')
    arrays = reconstruct(coordinates, frames=rules['frames'], bins=rules['distance_bins'],
                         distance_max_um=rules['distance_max_um'])
    after = [fingerprint(path) for path in protected_paths]
    require(before == after, 'Classification source changed during independent reconstruction')
    source_path = Path(__file__).resolve()
    source_text = source_path.read_text()
    unknown = group_counts['unknown']
    result = dict(sample=config.get('sample_name', config.get('sample_id', root.name)),
        source_selected_csv=str(csv_path), source_selected_csv_SHA256=before[2]['sha256'],
        source_classification_directory=str(root), classification_label_field=label_field,
        selected_count=selected_count, known_classified_count=selected_count - unknown,
        groups={name: group_counts[name] for name in GROUPS}, group_counts=group_counts,
        unknown=unknown, QC_excluded=summary['QC_excluded'], config_rules=rules,
        all_pair_population_name='knownclassified' if unknown else 'all_selected',
        all_pair_population_count=selected_count - unknown,
        all_pair_array_key_scope={'all_total': 'knownclassified' if unknown else 'all_selected',
                                  'all_background': 'knownclassified' if unknown else 'all_selected'},
        unknown_policy='Unknown rows counted separately and excluded from both classified groups and every pair array',
        all_histogram_shapes={name: list(array.shape) for name, array in arrays.items()},
        all_histogram_dtypes={name: str(array.dtype) for name, array in arrays.items()},
        all_histogram_C_order_bytes_SHA256={name: array_sha(array) for name, array in arrays.items()},
        hash_convention='SHA256 of complete little-endian int64 / float64 C-order numeric array bytes; no NPY header',
        independent_distance_histogram_implementation=True, production_distance_kernels_called=False,
        production_histograms_called=False, production_decomposition_called=False, point_estimates_computed=False,
        method='Python CSV round-trip floats; independently implemented NumPy broadcast dx/dy/sqrt and numpy.histogram; explicit frame/group loops',
        coordinate_conversion='native fit_xy_px * 117.0 / 1000 (multiplication then division), in micrometers',
        zero_rule='Cross-frame Cartesian pairs include exact zero; ordered same-frame pairs remove ALL exact zeros including coincident points',
        adjacency_rule='Only original frame f to original frame f+1; never compress empty frames or create gap/join pairs',
        histogram_bin_rule='numpy.histogram with linspace(0,3,301); left closed/right open except final right edge included',
        within_plus_cross_equals_all_knownclassified_all_bins_all_frames=True,
        background_within_plus_ordered_cross_equals_all_knownclassified_all_bins_all_frames=True,
        all2500_original_frames_and_empty_frames_preserved=True,
        empty_original_frame_indices_by_group={name: [frame for frame, rows in enumerate(coordinates[name]) if not len(rows)]
                                                for name in (*GROUPS, 'unknown')},
        same_frame_zero_removed_count={name: int(arrays['background_zero_removed'][group].sum())
                                       for group, name in enumerate(GROUPS)},
        support_bins=support_summaries(arrays, rules),
        inputs_before=before, inputs_after=after, source_fingerprints_unchanged=True,
        created_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=time.perf_counter() - started,
        source=dict(path=str(source_path), version=VERSION, sha256=file_sha(source_path), source_text=source_text),
        runtime=dict(python=sys.version, numpy=np.__version__, platform=platform.platform()),
        all_passed=True)
    if unknown == 0:
        result['within_plus_cross_equals_all_selected_all_bins_all_frames'] = True
        result['background_within_plus_ordered_cross_equals_all_selected_all_bins_all_frames'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--classification-dir', required=True, type=Path,
                        help='Completed postmin32 views root with resolved_config and classification/')
    parser.add_argument('--output', required=True, type=Path, help='Fresh JSON inside the project workspace')
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(PROJECT), 'Output must be inside the project workspace')
    require(not output.exists(), f'Output already exists: {output}')
    result = build_reference(args.classification_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(output), all_passed=True, selected_count=result['selected_count'],
        groups=result['group_counts'], histograms=result['all_histogram_C_order_bytes_SHA256']), indent=2))


if __name__ == '__main__':
    main()
