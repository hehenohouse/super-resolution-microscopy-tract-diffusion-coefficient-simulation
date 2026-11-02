"""Render only existing review/invalid 7x7 fits; never detect or fit points."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import shlex
import shutil
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import tifffile

from run_preview import draw_lib, dump, fingerprint, read_mat_vertices, roi_lib, video_and_checks


PURPLE = '#F28DE9'
ORANGE = '#FFAB40'
SCALE = 3
CROP = (82, 232, 347, 430)


def classify(table):
    valid = table.nominal_precision_valid.to_numpy(bool)
    clean = table.qc_unflagged_model_fit.to_numpy(bool)
    assert not (clean & ~valid).any()
    result = np.full(len(table), 'unflagged', dtype=object)
    result[valid & ~clean] = 'review'
    result[~valid] = 'unavailable'
    return result


def annotation(raw_shape, group, vertices, c, refined):
    """Same native pixel-center mapping as original overlays; no bound cap."""
    layer = draw_lib.overlay_layer(raw_shape, np.empty((0, 2)), vertices, c)
    d = ImageDraw.Draw(layer)
    ids = ImageFont.truetype(c['font_path'], 17)
    drawn = []
    for row in group.itertuples():
        review = row.problem_category == 'review'
        xy = [row.fit_x_px, row.fit_y_px] if refined and review else [row.x_px, row.y_px]
        assert np.isfinite(xy).all()
        x, y = draw_lib.display_center(xy, SCALE)
        if review:
            if refined:
                dx, dy = row.lpx * SCALE, row.lpy * SCALE
                assert np.isfinite([dx, dy]).all() and min(dx, dy) > 0
                d.line([(x-dx, y), (x+dx, y)], fill=PURPLE, width=1)
                d.line([(x, y-dy), (x, y+dy)], fill=PURPLE, width=1)
                d.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill=PURPLE)
            else:
                radius = 3 * SCALE
                d.ellipse((x-radius, y-radius, x+radius, y+radius), outline=PURPLE, width=2)
        else:
            radius = 3 * SCALE
            d.line([(x-radius, y-radius), (x+radius, y+radius)], fill=ORANGE, width=2)
            d.line([(x-radius, y+radius), (x+radius, y-radius)], fill=ORANGE, width=2)
        drawn.append({'spot_id': int(row.spot_id), 'category': row.problem_category,
                      'native_x_px': float(xy[0]), 'native_y_px': float(xy[1]),
                      'display_x_px': float(x), 'display_y_px': float(y)})
    if not refined:
        placed = []
        x0, y0, x1, y1 = (v*SCALE for v in CROP)
        for point in drawn:
            x, y = point['display_x_px'], point['display_y_px']
            text = str(point['spot_id']); text_width = d.textlength(text, font=ids)
            position = None
            # Offset only text; original floating marker centers never move.
            choices = [(12,-13),(12,12),(12,-38),(12,37),(-12-text_width,-13),
                       (-12-text_width,12),(-text_width/2,-40),(-text_width/2,28)]
            for dx, dy in choices:
                tx = max(x0+3, min(x+dx, x1-text_width-4))
                ty = max(y0+3, min(y+dy, y1-24))
                bbox = d.textbbox((tx,ty),text,font=ids,stroke_width=1)
                overlap = any(not (bbox[2]+3 < b[0] or b[2]+3 < bbox[0] or
                                   bbox[3]+3 < b[1] or b[3]+3 < bbox[1]) for b in placed)
                covers_marker = any((bbox[0]-5 < q['display_x_px'] < bbox[2]+5 and
                                     bbox[1]-5 < q['display_y_px'] < bbox[3]+5) for q in drawn)
                if not overlap and not covers_marker:
                    position = (tx,ty,bbox); break
            assert position is not None, 'Could not place a readable spot ID'
            tx,ty,bbox = position
            color = PURPLE if point['category']=='review' else ORANGE
            if abs(tx-(x+12)) > .1 or abs(ty-(y-13)) > .1:
                end = (max(bbox[0],min(x,bbox[2])),max(bbox[1],min(y,bbox[3])))
                d.line([(x,y),end],fill=color,width=1)
            d.text((tx,ty),text,font=ids,fill=color,stroke_width=1,stroke_fill='#101317')
            point['label_bbox_display_px'] = list(bbox)
            placed.append(bbox)
    assert len(drawn) == len(group) and all(x['category'] != 'unflagged' for x in drawn)
    return layer, drawn


def render(raw, group, vertices, frame, total, c, out):
    low, high = c['resolved_display_range_adu']
    gray = Image.fromarray(draw_lib.display_gray(raw, low, high, SCALE)).convert('RGBA')
    settings = dict(c, display_scale=SCALE)
    original_layer, original_drawn = annotation(raw.shape, group, vertices, settings, False)
    refined_layer, refined_drawn = annotation(raw.shape, group, vertices, settings, True)
    panels = [gray.convert('RGB'), Image.alpha_composite(gray, original_layer).convert('RGB'),
              Image.alpha_composite(gray, refined_layer).convert('RGB')]
    crop = tuple(v*SCALE for v in CROP)
    width, height = crop[2]-crop[0], crop[3]-crop[1]
    canvas = Image.new('RGB', (3*width+57, height+210), '#101317')
    d = ImageDraw.Draw(canvas)
    font = lambda size: ImageFont.truetype(c['font_path'], size)
    review = int((group.problem_category == 'review').sum())
    invalid = int((group.problem_category == 'unavailable').sum())
    d.text((14, 12), f'T1 C2 | Original frame {frame:04d} | {frame*.02:.2f} s | TrackMate 0.28 um -> Picasso 7x7 | Flagged points only',
           font=font(27), fill='white')
    labels = ['Raw C2 single frame', f'Flagged TrackMate centers | Purple {review} / Orange {invalid}',
              f'Flagged MLE +/- nominal X/Y CRLB | {review} bounds']
    for i, (panel, label) in enumerate(zip(panels, labels)):
        left = 14+i*(width+14)
        d.text((left, 60), label, font=font(22), fill='white')
        canvas.paste(panel.crop(crop), (left, 104))
    details = [
        'Purple: existing requires-review fits | Orange X: unavailable precision, marked at original TrackMate center | Yellow outline: cell ROI',
        'Numbers are spot IDs. Middle circles have fixed radius 3 px for position marking; right crosses are uncapped +/- lpx/lpy model SD.',
        f'Fixed {low:g}-{high:g} ADU. {len(group)} flagged / {total} ROI candidates this frame. Manufacturer-nominal CRLB; source fits and QC unchanged.'
    ]
    for j, text in enumerate(details):
        assert d.textbbox((0, 0), text, font=font(19))[2] < canvas.width-28
        d.text((14, height+116+j*29), text, font=font(19), fill='#DCE3EB')
    meta = PngImagePlugin.PngInfo()
    for key, value in {'original_frame': frame, 'ROI_candidates': total, 'review_points': review,
                       'unavailable_points': invalid, 'displayed_points': len(group),
                       'green_unflagged_points_drawn': 0, 'calibration_status': 'manufacturer_nominal'}.items():
        meta.add_text(key, str(value))
    target = out/'frames'/f'frame_{frame:04d}.png'
    canvas.save(target, pnginfo=meta, compress_level=5)
    with Image.open(target) as check:
        assert int(check.info['review_points']) == review
        assert int(check.info['unavailable_points']) == invalid
        assert int(check.info['green_unflagged_points_drawn']) == 0
        np.testing.assert_array_equal(np.asarray(check.crop((14,104,14+width,104+height))),
                                      np.asarray(gray.convert('RGB').crop(crop)))
    assert {x['spot_id'] for x in original_drawn} == {x['spot_id'] for x in refined_drawn} == set(group.spot_id)
    return {'frame': frame, 'time_s': frame*.02, 'ROI_candidates': total,
            'review_points': review, 'unavailable_points': invalid, 'displayed_points': len(group),
            'source_pixel_sha256': hashlib.sha256(raw.tobytes()).hexdigest()}, original_drawn, refined_drawn


def run(cache):
    started = time.perf_counter()
    cache = cache.resolve()
    resolved = json.loads((cache/'resolved_parameters.json').read_text())
    c = resolved['TrackMate']
    assert resolved['config']['box'] == 7 and c['radius_um'] == .28
    start, stop = int(c['frame_start']), int(c['frame_stop_exclusive'])
    assert (start, stop) == (2000, 2051)
    assert c['pixel_size_nm'] == 117 and c['frame_interval_s'] == .02
    old_done = json.loads((cache/'COMPLETED.json').read_text())
    assert old_done['all_checks_passed'] and old_done['frames'] == stop-start
    out = cache/('problem_points_only_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False); (out/'frames').mkdir()
    print('OUTPUT_DIRECTORY:', out, flush=True)
    with (out/'run.log').open('w') as log:
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = draw_lib.Tee(old_out, log), draw_lib.Tee(old_err, log)
        try:
            protected = [Path(c['input_tiff']), Path(c['roi_file']), Path(c['mat_file']),
                         cache/'ROI_points_with_refinement.csv', cache/'all_detections_with_refinement.csv',
                         cache/'frame_manifest.csv', cache/'resolved_parameters.json', cache/'COMPLETED.json',
                         cache/'output_fingerprints.json']
            print('Checking read-only input fingerprints...', flush=True)
            before = [fingerprint(p) for p in protected]; dump(out/'source_fingerprints_before.json', before)
            assert before[0]['sha256'] == c['expected_tiff_sha256']
            assert before[1]['sha256'] == c['expected_roi_sha256']
            old_hashes = json.loads((cache/'output_fingerprints.json').read_text())
            for item in before[3:6]:
                assert item['sha256'] == old_hashes[Path(item['path']).name]
            table = pd.read_csv(io.BytesIO((cache/'ROI_points_with_refinement.csv').read_bytes()), float_precision='round_trip')
            assert len(table) == old_done['ROI_candidates'] == 1173
            categories = classify(table)
            problems = table.loc[categories != 'unflagged'].copy()
            problems['problem_category'] = categories[categories != 'unflagged']
            assert len(problems) == 136
            assert problems.problem_category.value_counts().to_dict() == {'review':122, 'unavailable':14}
            problems.to_csv(out/'problem_points.csv', index=False, float_format='%.17g')
            reread = pd.read_csv(out/'problem_points.csv', float_precision='round_trip')
            pd.testing.assert_frame_equal(reread[table.columns], table.loc[categories != 'unflagged'].reset_index(drop=True), check_exact=True)
            vertices = roi_lib.load_imagej_roi(c['roi_file'])
            mat_vertices, _ = read_mat_vertices(c['mat_file'], c['mat_record'])
            np.testing.assert_array_equal(vertices, mat_vertices); assert len(vertices) == 454
            np.testing.assert_array_equal(roi_lib.points_in_polygon(table[['x_px','y_px']].to_numpy(), vertices), table.roi_inside.to_numpy()==1)
            config = {'source_cache':str(cache), 'original_frames':[start,stop-1],
                      'selection':'nominal_precision_valid AND NOT qc_unflagged_model_fit = review; NOT nominal_precision_valid = unavailable',
                      'middle_positions':'unchanged floating TrackMate centers',
                      'right_review_positions':'unchanged floating Picasso fitted centers',
                      'unavailable_positions':'original TrackMate centers; fitted coordinates not used',
                      'review_color':PURPLE,'unavailable_color':ORANGE,'ROI_outline':'yellow',
                      'circle_radius_px':3,'right_CRLB':'uncapped +/- lpx/lpy in native pixels',
                      'display_scale':SCALE,'display_crop_XYXY_px':CROP,
                      'display_range_ADU':c['resolved_display_range_adu'],'calibration_status':'manufacturer_nominal',
                      'redetect':False,'refit':False,'QC_changed':False,'video_fps':10}
            dump(out/'config.json', config)
            dump(out/'software_versions.json', {**{n:importlib.metadata.version(n) for n in ['numpy','pandas','Pillow','tifffile','h5py','PyYAML']},'python':sys.version,'executable':sys.executable})
            snapshot = out/'source_snapshot'; snapshot.mkdir()
            for src in [Path(__file__),Path(__file__).with_name('run_preview.py'),Path(draw_lib.__file__),Path(roi_lib.__file__)]:
                shutil.copyfile(src,snapshot/src.name)
            (out/'run_command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--cache',str(cache)])+'\n')
            reference_manifest = pd.read_csv(cache/'frame_manifest.csv')
            manifest, drawings = [], []
            with tifffile.TiffFile(c['input_tiff']) as tif:
                assert tif.series[0].axes == 'TYX' and tif.series[0].shape == (2500,684,428)
                for frame in range(start,stop):
                    group = problems[problems.frame == frame]
                    raw = tif.pages[frame].asarray(); assert raw.dtype == np.uint16
                    record, log_xy, mle_xy = render(raw,group,vertices,frame,int((table.frame==frame).sum()),c,out)
                    expected = reference_manifest[reference_manifest.frame==frame].iloc[0]
                    assert record['source_pixel_sha256'] == expected.source_pixel_sha256
                    assert record['ROI_candidates'] == expected.ROI_candidates
                    manifest.append(record); drawings.append({'frame':frame,'middle':log_xy,'right':mle_xy})
            counts = pd.DataFrame(manifest)
            assert counts.frame.tolist() == list(range(start,stop))
            assert int(counts.review_points.sum()) == 122 and int(counts.unavailable_points.sum()) == 14
            counts.to_csv(out/'frame_manifest.csv',index=False)
            dump(out/'drawn_points_audit.json',drawings)
            print('Rendered 51 frames: 122 review, 14 unavailable, zero unflagged markers.',flush=True)
            video, video_checks = video_and_checks(out,c)
            after = [fingerprint(p) for p in protected]; dump(out/'source_fingerprints_after.json',after)
            assert before == after, 'Protected source changed'
            validation = {'all_checks_passed':True,'ROI_MAT_454_vertices_exact':True,
                          'source_frames_pixel_fingerprints_all_exact':51,'all_raw_display_panels_exact':True,
                          'selected_original_fields_roundtrip_exact':True,'green_markers_drawn':0,
                          'frame_counts_match':True,'empty_problem_frames_preserved':counts.loc[counts.displayed_points==0,'frame'].tolist(),
                          'protected_sources_unchanged':True,'video_checks':video_checks,
                          'coordinate_rule':'(native + 0.5) * 3 - 0.5; X column, Y row down; no coordinate transform'}
            dump(out/'validation.json', validation)
            (out/'REPORT.txt').write_text('Existing TrackMate radius 0.28 um / Picasso 7x7 ROI cache, original frames 2000-2050 inclusive.\n'
                'Only 122 review and 14 unavailable points marked; 1037 unflagged points remain in source data but receive no marker here.\n'
                'Left: raw single frame. Middle: original TrackMate positions, purple circles / orange unavailable X, labeled with spot ID.\n'
                'Right: review MLE centers with uncapped nominal +/- lpx/lpy; unavailable X remains at TrackMate center.\n'
                'All 122 review points have existing suspect_psf_width flag; no new QC decisions made.\n'
                'Circles are radius-3-px position markers. CRLB uses manufacturer nominal calibration; no validated absolute precision.\n'
                '51 PNGs and H.264 10fps/5.1s MP4. Fixed 406-554 ADU, same cell crop and original image direction.\n'
                'Only rendering; no detection, fitting, trajectory linking, masking or TARDIS. Original sources unchanged.\n')
            outputs = list((out/'frames').glob('*.png'))+[out/'problem_points.csv',out/'frame_manifest.csv',video]
            dump(out/'output_fingerprints.json',{str(q.relative_to(out)):draw_lib.sha256(q) for q in outputs})
            dump(out/'COMPLETED.json',{'all_checks_passed':True,'output_directory':str(out),'frames':51,
                                    'review_points':122,'unavailable_points':14,'video':str(video),
                                    'elapsed_s':time.perf_counter()-started})
            print('COMPLETE:',out,flush=True)
        except Exception as exc:
            dump(out/'FAILED.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
        finally:
            sys.stdout,sys.stderr = old_out,old_err
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    run(parser.parse_args().cache)
