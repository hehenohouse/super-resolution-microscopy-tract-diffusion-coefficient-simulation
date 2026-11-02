"""Read stored MAT settings and compare three matched TIFF pages without tracking.

The two interpretations of the stored radius are hypotheses, not a calibration
claim. The original MAT, TIFF, ROI and preview are only read.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DATA = Path('/Volumes/My Book/Xinran_ComparisonData/20251016_TDKd3_HaloTRAK1_prcs - Copy/101625_TDKd3_TRAK1_T1f')
MAT = Path('/Volumes/My Book/Xinran_ComparisonData/TRAK1/TRAK1_TDK/20251016_TDK_TRAK1_V7.mat')
RECORD = 'C2-101625_TDKd3_TRAK1_T1'
FRAMES = [200, 224, 249]
HYPOTHESES = [
    ('current_3px', .351),
    ('stored_0p25_as_um', .25),
    ('stored_0p25_as_px', .25 * .117),
]


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def fingerprint(path):
    stat = path.stat()
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'sha256': h.hexdigest()}


def native(out):
    import numpy as np
    import imagej
    import scyjava
    from run_detection import make_imageplus, detect

    os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home'
    scyjava.config.set_java_constraints(fetch='never')
    scyjava.config.add_option('-Xmx2g')
    scyjava.config.add_option('-Dimagej.updater.disableAutocheck=true')
    ij = imagej.init('/Applications/Fiji.app', mode='headless')
    try:
        movie = np.load(out / 'selected_original_pages.npy')
        for name, radius in HYPOTHESES:
            imp = make_imageplus(ij, movie, .117, .02)
            rows, counts, version = detect(imp, radius, 6.5, True, 0, .02)
            assert version == '7.14.0', version
            for row in rows:
                row['frame'] = FRAMES[row['frame']]
                row['time_s'] = row['frame'] * .02
            with (out / f'{name}_all_detections.csv').open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [
                    'spot_id', 'frame', 'time_s', 'x_px', 'y_px', 'x_um', 'y_um',
                    'quality_log_response', 'radius_setting_um'])
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps({'hypothesis': name, 'radius_um': radius, 'frames': FRAMES, 'FOV_counts': counts}), flush=True)
        dump(out / 'native_versions.json', {'trackmate': version, 'imagej': str(ij.getVersion()),
             'java': str(scyjava.jimport('java.lang.System').getProperty('java.version'))})
    finally:
        ij.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-output', type=Path)
    args = parser.parse_args()
    if args.native_output:
        native(args.native_output)
        return
    import h5py
    import numpy as np
    import pandas as pd
    import tifffile
    from scipy.spatial import cKDTree
    sys.path.insert(0, str(PROJECT / 'TARDIS/path_irrelevant_python'))
    sys.path.insert(0, str(PROJECT / 'localization_overlays'))
    from tardis_tau1.locbio import points_in_polygon, load_imagej_roi
    from render_frames import display_gray, overlay_layer
    from PIL import Image, ImageDraw, ImageFont

    tiff = DATA / 'C2-101625_TDKd3_TRAK1_T1.tiff'
    roi = DATA / '0063-0331-0214.roi'
    out = DATA / 'analysis_results/trackmate_mat_parameter_comparison' / (
        'T1_C2_mat_settings_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=False)
    protected_before = [fingerprint(p) for p in [MAT, tiff, roi]]
    dump(out / 'source_fingerprints_before.json', protected_before)
    # Copy bytes only: external exFAT volumes do not support macOS chflags.
    shutil.copyfile(__file__, out / 'compare_mat_parameters.py')
    shutil.copyfile(ROOT / 'run_detection.py', out / 'run_detection.py')
    shutil.copyfile(ROOT / 'requirements.lock.txt', out / 'detector_requirements.lock.txt')
    (out / 'command.txt').write_text(shlex.join(sys.argv) + '\n')
    config = {'record': RECORD, 'frames': FRAMES, 'threshold': 6.5, 'median_filter': True,
              'subpixel_localization': True, 'hypotheses': HYPOTHESES,
              'pixel_size_um': .117, 'frame_interval_s': .02,
              'selection': 'Floating centers inside or on original ROI, tolerance 1e-9',
              'position_comparison': 'Mutual nearest same-frame detections, distances in original pixels',
              'warning': 'Stored radius unit and complete historical calibration/filter/export pipeline are unresolved.'}
    dump(out / 'config.json', config)
    vertices = load_imagej_roi(roi)
    def text(dataset):
        return ''.join(chr(int(c)) for c in dataset[()].ravel() if c)
    def fields(group):
        result = {}
        for key, item in group.items():
            if isinstance(item, h5py.Group):
                result[key] = fields(item)
            elif item.attrs.get('MATLAB_class') == b'char':
                result[key] = text(item)
            else:
                result[key] = item[()].tolist()
        return result
    with h5py.File(MAT, 'r') as handle:
        matches = [handle[ref] for ref in handle['data'][()].ravel() if text(handle[ref]['name']) == RECORD]
        assert len(matches) == 1
        record = matches[0]
        np.testing.assert_array_equal(vertices, record['roiData/roiCoordinates'][()].T)
        times = record['time'][()].ravel()
        frames = np.rint(times / .02).astype(int)
        np.testing.assert_allclose(frames * .02, times, atol=1e-8, rtol=0)
        old = pd.DataFrame({'frame': frames, 'x_px': record['x_data'][()].ravel(), 'y_px': record['y_data'][()].ravel()})
        metadata = {'record_path': record.name, 'settings': fields(record['trackMateResults/settingsTrackMate']),
                    'tracks_metadata': fields(record['tracksMetaData']),
                    'historical_source_tiff': text(record['sptData/fullPath']),
                    'stored_export_paths': {k: text(record['trackMateResults'][k]) for k in ['spotsSavePath', 'tracksSavePath']}}
    old['roi_inside'] = points_in_polygon(old[['x_px', 'y_px']].to_numpy(), vertices)
    old.loc[old.frame.isin(FRAMES)].to_csv(out / 'mat_detections_selected_frames.csv', index=False)
    metadata['counts'] = {'all': len(old), 'inside_common_float_ROI': int(old.roi_inside.sum()),
                         'outside_common_float_ROI': int((~old.roi_inside).sum()),
                         'frames_min_max': [int(old.frame.min()), int(old.frame.max())]}
    dump(out / 'mat_stored_parameters.json', metadata)
    with tifffile.TiffFile(tiff) as handle:
        assert len(handle.pages) == 2500
        movie = np.stack([handle.pages[f].asarray() for f in FRAMES])
        contrast_pixels = np.concatenate([handle.pages[f].asarray().ravel() for f in range(50)])
        lo, hi = np.percentile(contrast_pixels[contrast_pixels != 0], [1, 99.9])
        dump(out / 'tiff_metadata.json', {'shape_TYX': [2500, *movie.shape[1:]], 'dtype': str(movie.dtype),
              'imagej_metadata': handle.imagej_metadata, 'display_ADU': [float(lo), float(hi)]})
    assert movie.shape == (3, 684, 428) and movie.dtype == np.uint16
    np.save(out / 'selected_original_pages.npy', movie)
    command = [str((ROOT / '.venv/bin/python').absolute()), '-B', '-u', str(Path(__file__).resolve()), '--native-output', str(out)]
    (out / 'native_command.txt').write_text(shlex.join(command) + '\n')
    with (out / 'native_detection.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    selected = {}
    metrics = []
    for name, radius in HYPOTHESES:
        detected = pd.read_csv(out / f'{name}_all_detections.csv')
        xy = detected[['x_px', 'y_px']].to_numpy()
        finite = np.isfinite(xy).all(axis=1)
        inside = np.zeros(len(detected), bool)
        inside[finite] = points_in_polygon(xy[finite], vertices)
        detected['roi_inside'] = inside
        detected['position_determined'] = finite
        detected.to_csv(out / f'{name}_all_detections.csv', index=False)
        selected[name] = detected[inside]
        for f in FRAMES:
            a = old.loc[(old.frame == f) & old.roi_inside, ['x_px', 'y_px']].to_numpy()
            b = selected[name].loc[selected[name].frame == f, ['x_px', 'y_px']].to_numpy()
            if len(a) and len(b):
                da, ia = cKDTree(b).query(a)
                db, ib = cKDTree(a).query(b)
                mutual = np.flatnonzero(ib[ia] == np.arange(len(a)))
                matched = int(np.count_nonzero(da[mutual] <= .1))
                median = float(np.median(da))
            else:
                matched, median = 0, None
            metrics.append({'hypothesis': name, 'radius_um': radius, 'radius_px': radius / .117,
                'frame': f, 'mat_all_stored_points': int((old.frame == f).sum()), 'mat_common_ROI_points': len(a),
                'new_FOV_points': int((detected.frame == f).sum()), 'new_common_ROI_points': len(b),
                'mutual_pairs_within_0p1px': matched, 'fraction_mat_matched': matched / len(a) if len(a) else None,
                'fraction_new_matched': matched / len(b) if len(b) else None, 'mat_to_new_nearest_median_px': median})
    pd.DataFrame(metrics).to_csv(out / 'same_frame_comparison.csv', index=False)
    selected['MAT'] = old[old.roi_inside]
    draw_config = {'display_scale': 3, 'annotation_supersampling': 4, 'roi_color': '#FFD83D',
                   'roi_width_source_px': .5, 'circle_radius_source_px': 3, 'circle_width_source_px': .55,
                   'circle_color': '#00DCEB'}
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 20)
    # Crop display only to the same cell region; native coordinates stay unchanged.
    crop_native = (82, 232, 347, 430)
    labels = [('Raw C2', None), ('Original MAT', 'MAT'), ('Current radius 3 px', 'current_3px'),
              ('Hypothesis: 0.25 um = 2.14 px', 'stored_0p25_as_um')]
    crop = tuple(int(v * 3) for v in crop_native)
    panel_w, panel_h = crop[2] - crop[0], crop[3] - crop[1]
    for local_f, f in enumerate(FRAMES):
        raw = movie[local_f]
        plain = Image.fromarray(display_gray(raw, lo, hi, 3)).convert('RGBA')
        canvas = Image.new('RGB', (2 * panel_w + 30, 2 * (panel_h + 60) + 45), '#101317')
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 8), f'T1 C2 original frame {f} | {f*.02:.2f} s | Same raw page / common floating ROI', font=font, fill='white')
        for index, (label, key) in enumerate(labels):
            x = 10 + (index % 2) * (panel_w + 10)
            y = 43 + (index // 2) * (panel_h + 60)
            if key is None:
                panel = plain
                count_label = ''
            else:
                centers = selected[key].loc[selected[key].frame == f, ['x_px', 'y_px']].to_numpy()
                panel = Image.alpha_composite(plain, overlay_layer(raw.shape, centers, vertices, draw_config))
                count_label = f' | {len(centers)} ROI points'
            draw.text((x, y), label + count_label, font=font, fill='#DCE3EB')
            canvas.paste(panel.crop(crop).convert('RGB'), (x, y + 32))
        canvas.save(out / f'comparison_frame_{f:04d}.png')
    protected_after = [fingerprint(p) for p in [MAT, tiff, roi]]
    dump(out / 'source_fingerprints_after.json', protected_after)
    assert protected_before == protected_after
    dump(out / 'summary.json', {'output': str(out), 'mat': metadata, 'same_frame_comparison': metrics,
         'validation': {'ROI_vertices_exact_match': True, 'original_times_verified': True,
                        'source_fingerprints_unchanged': True, 'all_detected_candidates_preserved': True},
         'interpretation': 'Radius tests are diagnostic hypotheses. Matching does not establish historical radius units or all export/filter settings.'})
    (out / 'COMPLETED').write_text('Read-only source comparison completed. No tracking, Picasso fitting or TARDIS.\n')
    print(json.dumps({'output': str(out), 'same_frame_comparison': metrics}, indent=2), flush=True)


if __name__ == '__main__':
    main()
