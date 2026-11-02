"""Annotate frozen TrackMate candidates with Picasso's native net-gradient score.

This hybrid score check does not rerun either detector or Gaussian fitting. It
does not certify emitters. Original subpixel centers and all candidate rows are
preserved; an integer anchor is used only to evaluate the configured score window.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from picasso import localize
import tifffile
import yaml

from run_preview import draw_lib, dump, fingerprint, video_and_checks

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent


def radial_vectors(box):
    half = box // 2
    ux = np.zeros((box, box), np.float32)
    uy = np.zeros((box, box), np.float32)
    for i in range(box):
        ux[:, i] = uy[i, :] = half - i
    norm = np.sqrt(ux**2 + uy**2)
    # The center is excluded by the native scoring loop, as in Picasso.
    with np.errstate(divide='ignore', invalid='ignore'):
        ux /= norm
        uy /= norm
    return uy, ux


def validate_score(box, uy, ux):
    yy, xx = np.mgrid[:41, :47]
    image = np.rint(400 + 120 * np.exp(-((xx-22)**2+(yy-19)**2)/2)).astype(np.int32)
    y, x, score = localize.identify_in_image(image, -np.inf, box)
    native = localize._net_gradient(image, y, x, box, uy, ux)
    np.testing.assert_allclose(native, score, atol=0, rtol=0)
    a = localize._net_gradient(image, np.array([19]), np.array([22]), box, uy, ux)
    b = localize._net_gradient(image+500, np.array([19]), np.array([22]), box, uy, ux)
    c = localize._net_gradient(np.full_like(image, 400), np.array([19]), np.array([22]), box, uy, ux)
    assert a[0] > 0 and c[0] == 0
    np.testing.assert_array_equal(a, b)
    return {'native_identify_score_agreement': True, 'constant_background_invariant': True,
            'constant_patch_score_zero': True, 'synthetic_bright_spot_score': float(a[0])}


def render(raw, all_xy, pass_xy, vertices, frame, low, high, config, threshold, box=7):
    gray = Image.fromarray(draw_lib.display_gray(raw, low, high, 2)).convert('RGBA')
    panels = [Image.alpha_composite(gray, draw_lib.overlay_layer(raw.shape, xy, vertices, config)).convert('RGB')
              for xy in [all_xy, pass_xy]]
    margin, gap, header, footer = 24, 28, 104, 82
    width, height = panels[0].size
    image = Image.new('RGB', (2*margin+2*width+gap, header+height+footer), '#101317')
    image.paste(panels[0], (margin, header))
    image.paste(panels[1], (margin+width+gap, header))
    draw = ImageDraw.Draw(image)
    big = ImageFont.truetype(config['font_path'], 28)
    small = ImageFont.truetype(config['font_path'], 20)
    detail = ImageFont.truetype(config['font_path'], 17)
    draw.text((margin, 12), f'T1 C2 | Frame {frame:04d} | {frame*.02:.2f} s | ROI: {len(all_xy)} TrackMate -> {len(pass_xy)} gradient-pass', font=big, fill='white')
    draw.text((margin, 62), 'MAT settings: all TrackMate ROI detections', font=small, fill='#DCE3EB')
    draw.text((margin+width+gap, 62), f'Same frame + {box}x{box} net-gradient-pass centers', font=small, fill=config['circle_color'])
    y = header+height+8
    draw.text((margin, y), f'Picasso {importlib.metadata.version("picassosr")} native score | {box}x{box} | Strict net gradient > {threshold:.6f} | Raw unfiltered ADU', font=detail, fill='#DCE3EB')
    draw.text((margin, y+24), 'Frozen TrackMate floating centers; integer anchors for score only. Cyan radius-3-px circles mark position, NOT uncertainty.', font=detail, fill='#DCE3EB')
    draw.text((margin, y+48), f'Exploratory score gate; NOT proof of a bead / single emitter. No Gaussian fitting. Display {low:g}-{high:g} ADU. 10 fps slow playback.', font=detail, fill='#DCE3EB')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trackmate-preview', type=Path, required=True)
    parser.add_argument('--threshold-config', type=Path, default=PROJECT/'picasso_full_sensitivity/config.yaml')
    parser.add_argument('--box', type=int, choices=[5,7], default=7)
    parser.add_argument('--threshold-factor', type=float, default=1.0)
    args = parser.parse_args()
    source = args.trackmate_preview.resolve()
    assert (source/'COMPLETED.json').exists()
    config = yaml.safe_load((source/'resolved_config.yaml').read_text())
    assert config['radius_um'] == .25 and config['threshold'] == 6.5
    assert config['frame_start'] == 0 and config['frame_stop_exclusive'] == 50
    reference = yaml.safe_load(args.threshold_config.read_text())
    box = args.box
    if 'groups' in reference:
        group = [g for g in reference['groups'] if g['box'] == box and g['factor'] == 1]
        assert len(group) == 1
        reference_threshold = float(group[0]['threshold'])
    else:
        assert reference['box_size'] == box and reference['threshold_factor'] == 1
        reference_threshold = float(reference['absolute_threshold'])
    assert np.isfinite(args.threshold_factor) and args.threshold_factor > 0
    threshold = reference_threshold * args.threshold_factor
    half, margin = box//2, box//2+1
    score_column = f'picasso_net_gradient_box{box}'
    maximum_column = f'raw_local_maximum_box{box}'
    uy, ux = radial_vectors(box)
    checks = validate_score(box, uy, ux)
    factor_label = '' if args.threshold_factor == 1 else f'_factor{args.threshold_factor:.2f}'.replace('.', 'p')
    out = source.parents[1]/'trackmate_picasso_gradient_preview'/(
        f'T1_C2_MATsettings_box{box}_gradient{factor_label}_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=False)
    print('OUTPUT_DIRECTORY:', out, flush=True)
    original_csv = source/'all_detections.csv'
    movie_path = next((source/'native_detection').glob('*/input_frames.npy'))
    protected = [original_csv, source/'roi_detections.csv', source/'frame_manifest.csv',
                 source/'frames_0000_0049_10fps.mp4', movie_path, args.threshold_config]
    before = [fingerprint(p) for p in protected]
    dump(out/'source_fingerprints_before.json', before)
    shutil.copyfile(__file__, out/'run_gradient_preview.py')
    shutil.copyfile(args.threshold_config, out/'threshold_source_config.yaml')
    shutil.copyfile(ROOT/'run_preview.py', out/'dependency_run_preview.py')
    shutil.copyfile(PROJECT/'localization_overlays/render_frames.py', out/'dependency_render_frames.py')
    (out/'run_command.txt').write_text(shlex.join([sys.executable, '-B', '-u', str(Path(__file__).resolve()), *sys.argv[1:]])+'\n')
    dump(out/'config.json', {'trackmate_preview': str(source), 'frames': [0,49], 'box': box,
        'threshold': threshold, 'threshold_unit': 'ADU gradient sum', 'threshold_source': str(args.threshold_config),
        'reference_absolute_threshold': reference_threshold, 'threshold_factor': args.threshold_factor,
        'threshold_reestimated': False, 'coordinate_anchor': 'floor(TrackMate x/y + 0.5); original floating centers preserved',
        'gradient_implementation': 'Installed picasso.localize._net_gradient',
        'selection': 'ROI inside/boundary and finite score strictly above threshold',
        'raw_local_maximum_is_diagnostic_only': True, 'zero_support_is_diagnostic_only': True,
        'native_Picasso_detector_reexecuted_on_real_movie': False, 'fitting': False,
        'warning': 'Hybrid score evaluation at TrackMate anchors differs from complete native Picasso identification; not a truth classifier.'})
    table = pd.read_csv(original_csv, float_precision='round_trip')
    columns_before = list(table.columns)
    original = table.copy(deep=True)
    movie = np.load(movie_path, mmap_mode='r')
    assert movie.shape == (50,684,428) and movie.dtype == np.uint16
    xy = table[['x_px','y_px']].to_numpy()
    finite = np.isfinite(xy).all(axis=1)
    anchors = np.zeros(xy.shape, np.int64)
    anchors[finite] = np.floor(xy[finite]+.5).astype(np.int64)
    table['gradient_anchor_x_px'] = anchors[:,0]
    table['gradient_anchor_y_px'] = anchors[:,1]
    table[score_column] = np.nan
    table['gradient_support_valid'] = False
    table[maximum_column] = False
    table['source_zero_pixels_gradient_support'] = 0
    support = np.ones((box+2,box+2), bool)
    support[0,0] = support[0,-1] = support[-1,0] = support[-1,-1] = False
    source_pixels_verified = []
    with tifffile.TiffFile(config['input_tiff']) as tif:
        for frame in range(50):
            raw = np.asarray(movie[frame])
            np.testing.assert_array_equal(raw, tif.pages[frame].asarray())
            source_pixels_verified.append({'frame': frame, 'sha256': hashlib.sha256(raw.tobytes()).hexdigest()})
            idx = table.index[table.frame == frame].to_numpy()
            a = anchors[idx]
            valid = finite[idx] & (a[:,0]>=margin) & (a[:,1]>=margin) & (a[:,0]<428-margin) & (a[:,1]<684-margin)
            use = idx[valid]
            score = localize._net_gradient(raw.astype(np.int32), anchors[use,1], anchors[use,0], box, uy, ux)
            table.loc[use, score_column] = score
            table.loc[use, 'gradient_support_valid'] = True
            for i in use:
                x,y = anchors[i]
                patch = raw[y-half:y+half+1,x-half:x+half+1]
                table.loc[i, maximum_column] = np.argmax(patch) == box*box//2
                table.loc[i, 'source_zero_pixels_gradient_support'] = int(((raw[y-margin:y+margin+1,x-margin:x+margin+1] == 0)&support).sum())
            if (frame+1)%10 == 0: print(f'Scored frames {frame+1}/50', flush=True)
    table['gradient_support_has_zero'] = table.source_zero_pixels_gradient_support > 0
    table['net_gradient_pass'] = table.gradient_support_valid & np.isfinite(table[score_column]) & (table[score_column] > threshold)
    table['gradient_status'] = np.where(~table.gradient_support_valid, 'unavailable_support_or_position',
                                       np.where(table.net_gradient_pass, 'passes_score_threshold', 'below_or_equal_score_threshold'))
    table['net_gradient_threshold'] = threshold
    pd.testing.assert_frame_equal(original, table[columns_before], check_exact=True)
    table.to_csv(out/'all_detections_with_gradient.csv', index=False, float_format='%.17g')
    selected = table[(table.roi_inside == 1)&table.net_gradient_pass]
    selected.to_csv(out/'roi_gradient_pass.csv', index=False, float_format='%.17g')
    idx = pd.Index(range(50), name='frame')
    counts = pd.DataFrame(index=idx)
    roi_points = table[table.roi_inside == 1]
    for name, subset in [('all_ROI_TrackMate',roi_points),('ROI_gradient_pass',selected),
                         ('ROI_below_threshold',roi_points[roi_points.gradient_support_valid & ~roi_points.net_gradient_pass]),
                         ('ROI_unavailable',roi_points[~roi_points.gradient_support_valid]),
                         ('ROI_pass_with_zero_support',selected[selected.gradient_support_has_zero])]:
        counts[name] = subset.groupby('frame').size().reindex(idx,fill_value=0)
    np.testing.assert_array_equal(counts.all_ROI_TrackMate,counts[['ROI_gradient_pass','ROI_below_threshold','ROI_unavailable']].sum(axis=1))
    baseline_counts = pd.read_csv(source/'frame_counts.csv').set_index('frame')
    np.testing.assert_array_equal(counts.all_ROI_TrackMate, baseline_counts.roi_detections)
    counts['time_s'] = counts.index*.02
    counts.to_csv(out/'frame_counts.csv')
    vertices = pd.read_csv(source/'roi_vertices.csv')[['x_px','y_px']].to_numpy()
    low,high = config['resolved_display_range_adu']
    frames = out/'frames'; frames.mkdir()
    manifest = []
    for frame in range(50):
        all_xy = roi_points.loc[roi_points.frame == frame,['x_px','y_px']].to_numpy()
        pass_xy = selected.loc[selected.frame == frame,['x_px','y_px']].to_numpy()
        image = render(movie[frame], all_xy, pass_xy, vertices, frame, low, high, config, threshold, box)
        meta = PngImagePlugin.PngInfo()
        for k,v in {'original_frame':frame,'TrackMate_ROI_count':len(all_xy),'gradient_pass_ROI_count':len(pass_xy)}.items(): meta.add_text(k,str(v))
        path = frames/f'frame_{frame:04d}.png'
        image.save(path,pnginfo=meta,compress_level=5)
        with Image.open(path) as readback:
            assert int(readback.info['TrackMate_ROI_count']) == counts.loc[frame,'all_ROI_TrackMate']
            assert int(readback.info['gradient_pass_ROI_count']) == counts.loc[frame,'ROI_gradient_pass']
        manifest.append({'frame':frame,'time_s':frame*.02,'TrackMate_ROI_count':len(all_xy),'gradient_pass_ROI_count':len(pass_xy),**source_pixels_verified[frame]})
    pd.DataFrame(manifest).to_csv(out/'frame_manifest.csv',index=False)
    video,video_checks = video_and_checks(out,config)
    after = [fingerprint(p) for p in protected]
    dump(out/'source_fingerprints_after.json',after)
    assert before == after
    dump(out/'validation.json', {'all_checks_passed':True, 'original_candidates_and_fields_unchanged':True,
          'all_50_pages_pixel_exact':True, 'all_original_frame_numbers_retained':True,
          'counts_match_baseline':True, 'all_50_png_counts_match_csv':True, 'native_score_checks':checks,
          'source_preview_and_reference_fingerprints_unchanged':True, 'video_checks':video_checks})
    summary = {'output':str(out),'all_FOV_candidates_preserved':len(table),'ROI_TrackMate_points':len(roi_points),
               'ROI_gradient_pass':len(selected),'ROI_below_threshold':int(counts.ROI_below_threshold.sum()),
               'ROI_unavailable':int(counts.ROI_unavailable.sum()),'ROI_pass_with_zero_support':int(counts.ROI_pass_with_zero_support.sum()),
               'threshold_ADU_gradient':threshold,'box':box,'reference_absolute_threshold':reference_threshold,
               'threshold_factor':args.threshold_factor,'fit_executed':False,
               'versions':{k:importlib.metadata.version(k) for k in ['picassosr','numpy','pandas','Pillow','tifffile']}}
    dump(out/'summary.json',summary)
    (out/'README.txt').write_text(
        f'Frozen MAT-settings TrackMate candidates, frames 0-49; Picasso native {box}x{box} net-gradient score on unfiltered raw ADU.\n'
        f'Threshold {threshold:.17g} = previous Box{box} factor1 threshold {reference_threshold:.17g} x factor {args.threshold_factor:g}; not an instrument default or a validated truth classifier.\n'
        'Integer score anchors use floor(x/y+0.5); original TrackMate floating coordinates and every candidate are preserved.\n'
        'This hybrid applies the native score to TrackMate anchors. It does not rerun raw-local-maximum detection, relocate candidates or fit PSFs.\n'
        'Raw local-maximum and zero-support flags are diagnostics, not extra selection gates. Unavailable support remains explicit.\n'
        'Left: all TrackMate ROI circles. Right: same raw frame with only score-pass ROI circles. Same fixed 406-554 ADU display.\n'
        '50 PNGs; H264 50 frames / 10 fps / 5 seconds. Acquisition interval is 20ms. Circles are position markers, not errors.\n'
        'No photon estimates, localization errors, Gaussian-fit QC, tracking, TARDIS or mitochondrial analysis in this score trial.\n')
    dump(out/'COMPLETED.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
