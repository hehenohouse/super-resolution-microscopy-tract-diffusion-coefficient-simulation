"""Exact-frame scientific image overlays using frozen QC+ROI localization centers."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import tifffile
import yaml

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def fingerprint(path):
    path = Path(path)
    stat = path.stat()
    return {'path': str(path), 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'sha256': sha256(path)}


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.original.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.original.flush()
        self.log.flush()


def display_gray(raw, low, high, scale):
    # Intensity changes are display-only. Spatial enlargement uses exact pixel repetition.
    gray = np.rint(np.clip((raw.astype(np.float64) - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return np.repeat(np.repeat(gray, scale, axis=0), scale, axis=1)


def display_center(xy, scale):
    # Native pixel center 0 maps to the center of the first enlarged pixel block.
    return (np.asarray(xy, dtype=float) + 0.5) * scale - 0.5


def overlay_layer(shape, xy, vertices, c):
    height, width = shape
    scale, ss = c['display_scale'], c['annotation_supersampling']
    layer = Image.new('RGBA', (width * scale * ss, height * scale * ss), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    # Rasterize only annotations at high resolution. Raw image pixels are never smoothed.
    roi = display_center(vertices, scale)
    roi_hi = display_center(roi, ss)
    if len(roi_hi):
        draw.line([tuple(p) for p in np.vstack([roi_hi, roi_hi[0]])],
                  fill=c['roi_color'], width=max(1, round(c['roi_width_source_px'] * scale * ss)), joint='curve')
    centers = display_center(display_center(xy, scale), ss)
    radius = c['circle_radius_source_px'] * scale * ss
    stroke = max(1, round(c['circle_width_source_px'] * scale * ss))
    for x, y in centers:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     outline=c['circle_color'], width=stroke)
    return layer.resize((width * scale, height * scale), Image.Resampling.LANCZOS)


def render(raw, points, vertices, frame, low, high, c):
    gray = display_gray(raw, low, high, c['display_scale'])
    plain = Image.fromarray(gray).convert('RGB')
    marked = Image.alpha_composite(plain.convert('RGBA'),
                                  overlay_layer(raw.shape, points, vertices, c)).convert('RGB')
    margin, gap, header, footer = 24, 28, 104, 58
    width, height = plain.size
    canvas = Image.new('RGB', (2 * margin + 2 * width + gap, header + height + footer), '#101317')
    canvas.paste(plain, (margin, header))
    canvas.paste(marked, (margin + width + gap, header))
    draw = ImageDraw.Draw(canvas)
    large = ImageFont.truetype(c['font_path'], 28)
    small = ImageFont.truetype(c['font_path'], 20)
    detail = ImageFont.truetype(c['font_path'], 17)
    draw.text((margin, 12), f'T1 C2 | Frame {frame:04d} | {frame * c["frame_interval_s"]:.2f} s | {len(points)} QC ROI points',
              fill='white', font=large)
    draw.text((margin, 62), 'Original single frame (ADU display)', fill='#DCE3EB', font=small)
    draw.text((margin + width + gap, 62), 'Same frame + TRAK1 localization circles', fill=c['circle_color'], font=small)
    y = header + height + 8
    box_label = f'Box {c["box_size"]}'
    if c.get('experimental_even_box'):
        box_label = f'Fit Box {c["box_size"]} / Detect {c["detection_box_size"]} (experimental)'
    draw.text((margin, y), f'{box_label} | Threshold {c["threshold_factor"]:.2f} | Fixed display: {low:g}-{high:g} ADU | {c["pixel_size_nm"]:g} nm/pixel | Manufacturer nominal calibration',
              fill='#DCE3EB', font=detail)
    draw.text((margin, y + 24), 'Cyan circles: fitted centers, radius 3 pixels; NOT uncertainty. Yellow: cell ROI. Original coordinates; no temporal averaging.',
              fill='#DCE3EB', font=detail)
    return canvas, (margin, header, margin + width, header + height)


def coordinate_checks(c):
    # Asymmetric centers detect X/Y swap, vertical flip and premature integer rounding.
    vertices = np.empty((0, 2))
    raw_shape = (20, 28)
    xy = np.array([[8.25, 12.75]])
    mask = np.asarray(overlay_layer(raw_shape, xy, vertices, c))[:, :, 3]
    yy, xx = np.indices(mask.shape)
    mass = mask.sum()
    actual = np.array([(xx * mask).sum() / mass, (yy * mask).sum() / mass])
    expected = display_center(xy[0], c['display_scale'])
    np.testing.assert_allclose(actual, expected, atol=0.5, rtol=0)
    empty = np.asarray(overlay_layer(raw_shape, np.empty((0, 2)), vertices, c))
    assert not empty.any()
    return {'expected_display_center_xy': expected.tolist(), 'raster_alpha_centroid_xy': actual.tolist(),
            'maximum_tolerance_display_px': 0.5, 'empty_annotation_layer_ok': True,
            'native_integer_center_rule': '(native + 0.5) * display_scale - 0.5; Y grows downward'}


def encode_and_verify(out, c, count):
    video = out / f'frames_{c["frame_start"]:04d}_{c["frame_stop_exclusive"] - 1:04d}_50fps.mp4'
    command = [shutil.which('ffmpeg'), '-nostdin', '-hide_banner', '-loglevel', 'warning',
               '-framerate', str(c['video_fps']), '-start_number', str(c['frame_start']),
               '-i', str(out / 'frames/frame_%04d.png'), '-frames:v', str(count), '-an',
               '-c:v', 'libx264', '-preset', 'medium', '-threads', str(c['video_threads']),
               '-crf', str(c['video_crf']), '-pix_fmt', 'yuv420p', '-r', str(c['video_fps']),
               '-fps_mode', 'cfr', '-movflags', '+faststart', str(video)]
    (out / 'video_command.txt').write_text(shlex.join(command) + '\n')
    with (out / 'video_encode.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    probe_command = [shutil.which('ffprobe'), '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                     '-show_entries', 'stream=codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,nb_read_frames,duration:format=duration',
                     '-of', 'json', str(video)]
    probe = json.loads(subprocess.check_output(probe_command, text=True))
    dump(out / 'video_probe.json', probe)
    stream = probe['streams'][0]
    assert stream['codec_name'] == 'h264'
    assert int(stream['nb_read_frames']) == count == int(stream['nb_frames'])
    assert stream['r_frame_rate'] == stream['avg_frame_rate'] == f'{c["video_fps"]}/1'
    assert abs(float(probe['format']['duration']) - count / c['video_fps']) < 1e-6
    width, height = stream['width'], stream['height']
    checks = []
    for frame in [0, 124, 249]:
        command = [shutil.which('ffmpeg'), '-nostdin', '-v', 'error', '-i', str(video),
                   '-vf', f'select=eq(n\\,{frame})', '-frames:v', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
        decoded = subprocess.check_output(command)
        assert len(decoded) == width * height * 3
        image = np.frombuffer(decoded, np.uint8).reshape(height, width, 3)
        expected = np.asarray(Image.open(out / 'frames' / f'frame_{frame:04d}.png').convert('RGB'))
        mae = float(np.abs(image.astype(float) - expected.astype(float)).mean())
        assert mae < 8, (frame, mae)
        qa = out / 'validation_video_frames'
        qa.mkdir(exist_ok=True)
        Image.fromarray(image).save(qa / f'decoded_{frame:04d}.png')
        checks.append({'frame': frame, 'mean_absolute_RGB_error_0_255': mae,
                       'video_vs_PNG_check_passed': True})
    return video, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.yaml')
    args = parser.parse_args()
    cfg = Path(args.config).resolve()
    c = yaml.safe_load(cfg.read_text())
    assert c['frame_start'] == 0 and c['frame_stop_exclusive'] == 250 and c['video_fps'] == 50
    assert not any(c[x] for x in ['redetect', 'refit', 'tardis', 'mitochondrial_classification'])
    assert shutil.which('ffmpeg') and shutil.which('ffprobe')
    data = Path(c['data_directory'])
    assert data.is_dir(), 'External source disk must already be mounted.'
    source = (cfg.parent / c['source_roi_run']).resolve()
    tiff = data / c['source_tiff']
    locs_path = source / 'selected' / (c['source_group'] + '.csv')
    roi_path = source / 'roi_vertices.csv'
    roi_info = json.loads((source / 'roi_validation.json').read_text())
    assert roi_info['vertices_exactly_equal'] and (source / 'COMPLETED.json').exists()
    out = data / c['output_relative_directory'] / ('T1_C2_' + c['source_group'] + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=False)
    (out / 'frames').mkdir()
    log = (out / 'run.log').open('w')
    sys.stdout, sys.stderr = Tee(sys.stdout, log), Tee(sys.stderr, log)
    started = time.perf_counter()
    print('OUTPUT_DIRECTORY:', out, flush=True)
    try:
        protected = [tiff, locs_path, roi_path, source / 'config.yaml', source / 'roi_validation.json',
                     source / 'COMPLETED.json', source / 'selected' / (c['source_group'] + '.hdf5'),
                     source / 'audit' / (c['source_group'] + '.csv'), Path(roi_info['roi_source'])]
        before = [fingerprint(p) for p in protected]
        assert before[0]['sha256'] == roi_info['source_tiff_metadata']['source_sha256']
        assert before[-1]['sha256'] == roi_info['roi_sha256']
        dump(out / 'source_fingerprints_before.json', before)
        table = pd.read_csv(locs_path, float_precision='round_trip')
        selected = table.loc[table.frame.between(c['frame_start'], c['frame_stop_exclusive'] - 1)].copy()
        assert len(selected) == c['expected_localizations']
        assert (selected.qc_unflagged_model_fit & selected.roi_inside & selected.stats_selected_roi).all()
        assert (selected.calibration_status == c['calibration_status']).all()
        assert (selected.box_size == c['box_size']).all() and (selected.threshold_factor == c['threshold_factor']).all()
        np.testing.assert_allclose(selected.absolute_threshold, c['absolute_threshold'], rtol=0, atol=1e-10)
        assert np.isfinite(selected[['x', 'y']].to_numpy()).all()
        np.testing.assert_allclose(selected.time_s, selected.frame * c['frame_interval_s'], atol=1e-12)
        vertices = pd.read_csv(roi_path)[['x_px', 'y_px']].to_numpy(float)
        synthetic = coordinate_checks(c)
        groups = {int(f): g for f, g in selected.groupby('frame')}
        versions = {name: importlib.metadata.version(name) for name in ['Pillow', 'numpy', 'pandas', 'tifffile', 'PyYAML', 'picassosr']}
        versions.update(python=sys.version, python_executable=sys.executable, environment_prefix=sys.prefix,
                        platform=platform.platform(), ffmpeg=subprocess.check_output(['ffmpeg', '-version'], text=True).splitlines()[0])
        dump(out / 'software_versions.json', versions)
        # This external filesystem does not implement every macOS chflags attribute.
        # Archive exact bytes; source timestamps/flags are not required for reproduction.
        shutil.copyfile(__file__, out / 'render_frames.py')
        shutil.copyfile(cfg, out / 'config.yaml')
        assert sha256(__file__) == sha256(out / 'render_frames.py')
        assert sha256(cfg) == sha256(out / 'config.yaml')
        c['source_roi_run'] = str(source)
        (out / 'run_command.txt').write_text(f'cd {shlex.quote(str(cfg.parent))}\n' +
            shlex.join([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--config', str(cfg)]) + '\n')
        rows, annotations = [], []
        with tifffile.TiffFile(tiff) as tf:
            assert len(tf.pages) == 2500 and tf.pages[0].shape == (684, 428) and tf.pages[0].dtype == np.uint16
            info = tf.imagej_metadata
            assert info['frames'] == 2500 and info['finterval'] == c['frame_interval_s']
            assert info['spacing'] * 1000 == c['pixel_size_nm']
            labels = info['Labels']
            pooled = np.concatenate([tf.pages[f].asarray().ravel() for f in range(*c['contrast_frames'])])
            low, high = np.percentile(pooled[pooled != 0], c['contrast_percentiles'])
            del pooled
            assert high > low
            c['resolved_display_range_ADU'] = [float(low), float(high)]
            (out / 'resolved_config.yaml').write_text(yaml.safe_dump(c, sort_keys=False))
            print(f'Fixed display: {low:g}–{high:g} ADU; total {len(selected)} circles.', flush=True)
            for frame in range(c['frame_start'], c['frame_stop_exclusive']):
                assert f' t:{frame + 1}/2500 ' in labels[frame], (frame, labels[frame])
                raw = tf.pages[frame].asarray()
                group = groups.get(frame, selected.iloc[:0])
                xy = group[['x', 'y']].to_numpy(float)
                assert ((xy[:, 0] >= -0.5) & (xy[:, 0] <= raw.shape[1] - 0.5)).all()
                assert ((xy[:, 1] >= -0.5) & (xy[:, 1] <= raw.shape[0] - 0.5)).all()
                canvas, plain_box = render(raw, xy, vertices, frame, low, high, c)
                image_path = out / 'frames' / f'frame_{frame:04d}.png'
                meta = PngImagePlugin.PngInfo()
                meta.add_text('original_frame', str(frame))
                meta.add_text('time_s', f'{frame * c["frame_interval_s"]:.2f}')
                meta.add_text('roi_QC_point_count', str(len(xy)))
                meta.add_text('source_page_label', labels[frame])
                meta.add_text('circle_meaning', 'fixed-radius fitted-position marker; not uncertainty')
                canvas.save(image_path, pnginfo=meta, compress_level=5)
                # Independent readback validates every saved original-image panel and label.
                with Image.open(image_path) as saved:
                    readback = np.asarray(saved.crop(plain_box).convert('L'))
                    reference = ((np.minimum(np.maximum(raw.astype(float), low), high) - low) * (255 / (high - low))).round().astype(np.uint8)
                    reference = reference.repeat(c['display_scale'], 0).repeat(c['display_scale'], 1)
                    np.testing.assert_array_equal(readback, reference)
                    assert int(saved.info['original_frame']) == frame and int(saved.info['roi_QC_point_count']) == len(xy)
                pixel_hash = hashlib.sha256(raw.tobytes()).hexdigest()
                rows.append({'frame': frame, 'time_s': frame * c['frame_interval_s'], 'source_tiff_page': frame,
                             'roi_point_count': len(xy), 'png': str(image_path.relative_to(out)),
                             'source_pixel_sha256': pixel_hash, 'png_sha256': sha256(image_path),
                             'source_label': labels[frame], 'original_panel_verified': True})
                for point in group.itertuples():
                    dx, dy = display_center([point.x, point.y], c['display_scale'])
                    annotations.append({'frame': frame, 'candidate_id': int(point.candidate_id),
                                        'source_x_px': point.x, 'source_y_px': point.y,
                                        'right_panel_relative_x_px': dx, 'right_panel_relative_y_px': dy,
                                        'radius_source_px': c['circle_radius_source_px']})
                if (frame + 1) % 25 == 0:
                    print(f'PNG frames completed: {frame + 1}/250', flush=True)
        pd.DataFrame(rows).to_csv(out / 'frame_manifest.csv', index=False, float_format='%.17g')
        pd.DataFrame(annotations).to_csv(out / 'annotation_positions.csv', index=False, float_format='%.17g')
        assert len(rows) == 250 and sum(r['roi_point_count'] for r in rows) == len(annotations) == c['expected_localizations']
        assert len(list((out / 'frames').glob('frame_*.png'))) == 250
        print('Encoding video: 50 fps, 250 frames, 5 seconds.', flush=True)
        video, video_checks = encode_and_verify(out, c, len(rows))
        after = [fingerprint(p) for p in protected]
        assert before == after, 'A protected source changed during rendering.'
        dump(out / 'source_fingerprints_after.json', after)
        dump(out / 'validation.json', {'all_checks_passed': True, 'png_frames': 250, 'total_circles': len(annotations),
            'every_original_panel_matches_correct_source_page': True, 'every_frame_label_and_count_verified': True,
            'position_checks': synthetic, 'video_frame_checks': video_checks, 'original_sources_unchanged': True,
            'nominal_calibration_preserved': True, 'no_redetection_refit_TARDIS_or_mito_classification': True})
        dump(out / 'output_fingerprints.json', {'video_sha256': sha256(video),
            'frame_manifest_sha256': sha256(out / 'frame_manifest.csv'),
            'annotation_positions_sha256': sha256(out / 'annotation_positions.csv')})
        (out / 'README.txt').write_text(
            'T1 C2 original frames 0–249: original vs TRAK1 fitted-position markers.\n'
            '250 PNGs and H.264 MP4, 50fps, 5 seconds. Single frames, no temporal averaging.\n'
            f'Box{c["box_size"]}, threshold factor{c["threshold_factor"]:.2f}, absolute net-gradient{c["absolute_threshold"]!r}; existing QC AND cell ROI.\n'
            f'{len(annotations)} circles in total. Cyan radius3 source pixels is a position marker, NOT an error bar.\n'
            'Yellow contour is the unchanged common cell ROI. Native C2 X/Y; Y increases downward.\n'
            f'Fixed display gray range {low:g}–{high:g} ADU from pooled nonzero original frames0–49 p1/p99.9.\n'
            'Original16-bit TIFF remains unchanged. Display is8-bit gray, enlarged2x with nearest-neighbor pixel repetition.\n'
            'Only drawing layers are antialiased. MP4 is compressed for visual inspection, not pixel measurement.\n'
            'Manufacturer-nominal calibration retained; no refitting, tracking, TARDIS or mitochondria classification.\n'
            'Source paths/hashes: source_fingerprints_before/after.json. Parameters: resolved_config.yaml.\n'
            'Each PNG has original frame/time/count metadata; frame_manifest.csv records all source page/pixel fingerprints.\n'
            'annotation_positions.csv preserves source floating centers and corresponding panel coordinates.\n'
            'Exact command, source script, versions and validation are in this directory.\n'
            'The broader result-copy plan has NOT been executed.\n')
        dump(out / 'runtime.json', {'elapsed_seconds': time.perf_counter() - started, 'refitting_seconds': 0})
        dump(out / 'COMPLETED.json', {'frames': 250, 'circles': len(annotations), 'video_fps': 50, 'video_seconds': 5,
            'all_checks_passed': True, 'finished_utc': datetime.now(timezone.utc).isoformat(), 'output_directory': str(out)})
        print('COMPLETE:', out, flush=True)
    except Exception as exc:
        dump(out / 'FAILED.json', {'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
