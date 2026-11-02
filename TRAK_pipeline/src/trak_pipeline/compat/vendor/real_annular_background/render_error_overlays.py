"""Raw image versus current cached 7x7 annular-fit error bars; no refit."""
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
import tifffile
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'trackmate_python_bridge'))
from run_preview import draw_lib, dump, fingerprint, read_mat_vertices, roi_lib


def text_to_width(draw, xy, text, font_path, size, color, max_width):
    font = ImageFont.truetype(font_path,size)
    while draw.textbbox((0,0),text,font=font)[2] > max_width and size>10:
        size-=1
        font = ImageFont.truetype(font_path,size)
    draw.text(xy,text,font=font,fill=color)


def render(raw, group, vertices, preview, frame, out):
    scale = 3
    c = dict(preview,display_scale=scale)
    low,high = c['resolved_display_range_adu']
    gray = Image.fromarray(draw_lib.display_gray(raw,low,high,scale)).convert('RGBA')
    layer = draw_lib.overlay_layer(raw.shape,np.empty((0,2)),vertices,c)
    draw = ImageDraw.Draw(layer)
    nvalid = 0
    for row in group.itertuples():
        if row.histogram_eligible:
            x,y = draw_lib.display_center([row.fit_x_px,row.fit_y_px],scale)
            dx,dy = row.x_se_nm/117*scale,row.y_se_nm/117*scale
            assert np.isfinite([x,y,dx,dy]).all() and min(dx,dy)>0
            color = '#35E6A0' if row.qc_unflagged_model_fit else '#F28DE9'
            draw.line([(x-dx,y),(x+dx,y)],fill=color,width=1)
            draw.line([(x,y-dy),(x,y+dy)],fill=color,width=1)
            draw.ellipse((x-1.3,y-1.3,x+1.3,y+1.3),fill=color)
            nvalid+=1
        else:
            x,y = draw_lib.display_center([row.x_px,row.y_px],scale)
            draw.line([(x-3,y-3),(x+3,y+3)],fill='#FFAB40',width=1)
            draw.line([(x-3,y+3),(x+3,y-3)],fill='#FFAB40',width=1)
    assert nvalid==int(group.histogram_eligible.sum())
    overlay = Image.alpha_composite(gray,layer).convert('RGB')
    crop = tuple(scale*v for v in (82,232,347,430))
    w,h = crop[2]-crop[0],crop[3]-crop[1]
    header,margin,gap = 104,14,14
    canvas = Image.new('RGB',(2*w+2*margin+gap,h+222),'#101317')
    draw = ImageDraw.Draw(canvas)
    width = canvas.width-2*margin
    text_to_width(draw,(margin,12),
        f'T1 C2 | Original frame {frame:04d} | {frame*.02:.2f} s | TrackMate 0.28 um -> current 7x7 annular-background fit',
        c['font_path'],26,'white',width)
    labels = ['Raw C2 single frame',f'Current 7x7 | {nvalid} error bars / {len(group)} ROI candidates']
    for j,(panel,label) in enumerate(zip([gray.convert('RGB'),overlay],labels)):
        x = margin+j*(w+gap)
        text_to_width(draw,(x,60),label,c['font_path'],23,'white',w)
        canvas.paste(panel.crop(crop),(x,header))
    lines = [
        f'Green: unflagged | Magenta: overlap review | Orange X: no available error ({len(group)-nvalid}) | Yellow: cell ROI',
        'Crosses: +/-1 per-axis full-Hessian model standard error; nominal calibration, NOT validated accuracy or native Picasso CRLB.',
        f'Same single raw TIFF page; fixed {low:g}-{high:g} ADU. Cell crop is display-only. Cached centers, errors and QC unchanged.']
    for j,text in enumerate(lines):
        text_to_width(draw,(margin,header+h+12+j*30),text,c['font_path'],18,'#DCE3EB',width)
    info = dict(original_frame=frame,time_s=frame*.02,ROI_candidates=len(group),
        available_errors=nvalid,unflagged=int(group.qc_unflagged_model_fit.sum()),
        unavailable_errors=len(group)-nvalid,source_pixel_sha256=hashlib.sha256(raw.tobytes()).hexdigest())
    metadata = PngImagePlugin.PngInfo()
    for k,v in info.items():metadata.add_text(k,str(v))
    path = out/f'frame_{frame:04d}_raw_and_errorbars.png'
    canvas.save(path,pnginfo=metadata,compress_level=5)
    with Image.open(path) as check:
        assert int(check.info['available_errors'])==nvalid
        assert int(check.info['original_frame'])==frame
        np.testing.assert_array_equal(np.asarray(check.crop((margin,header,margin+w,header+h))),
            np.asarray(gray.convert('RGB').crop(crop)))
    return dict(**info,image=str(path),png_sha256=draw_lib.sha256(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--frames',nargs='+',type=int,default=[200,1200,2000])
    args = ap.parse_args()
    source = args.source.resolve(); frames = args.frames
    assert len(frames)==len(set(frames)) and all(0<=f<2500 for f in frames)
    assert json.loads((source/'COMPLETED.json').read_text())['all_checks_passed']
    rules = json.loads((source/'rules_before_run.json').read_text())
    assert rules['fit_rules']['box_px']==7 and rules['fit_rules']['pixel_size_nm']==117
    preview = yaml.safe_load((Path(rules['fit_rules']['source_preview'])/'resolved_config.yaml').read_text())
    inputs = [source/'all_detections.csv',source/'all_ROI_target_fits.csv',source/'roi_vertices.csv',
        Path(preview['roi_file']),Path(preview['mat_file'])]
    before = [fingerprint(p) for p in inputs]
    vertices = roi_lib.load_imagej_roi(preview['roi_file'])
    np.testing.assert_array_equal(vertices,pd.read_csv(inputs[2]).to_numpy())
    mat_vertices,_ = read_mat_vertices(preview['mat_file'],preview['mat_record'])
    np.testing.assert_array_equal(vertices,mat_vertices)
    assert len(vertices)==454 and before[3]['sha256']==preview['expected_roi_sha256']
    # Only read/filter caches. No fitting or detection functions are imported.
    detections = pd.read_csv(inputs[0],float_precision='round_trip')
    fits = pd.read_csv(inputs[1],low_memory=False,float_precision='round_trip')
    detections = detections[detections.frame.isin(frames)].copy()
    fits = fits[fits.frame.isin(frames)].copy()
    targets = detections[detections.roi_inside.eq(1)]
    np.testing.assert_array_equal(fits.candidate_index,targets.candidate_index)
    np.testing.assert_allclose(fits.time_s,fits.frame*.02,rtol=0,atol=1e-12)
    np.testing.assert_allclose(fits.x_se_nm/117,fits.lpx_model_px,equal_nan=True)
    np.testing.assert_allclose(fits.y_se_nm/117,fits.lpy_model_px,equal_nan=True)
    table = fits.join(targets.set_index('candidate_index')[['x_px','y_px','source_detection_uid']],on='candidate_index')
    pd.testing.assert_frame_equal(table[fits.columns],fits)
    TIFF = Path(preview['input_tiff'])
    source_fingerprint = next(x for x in json.loads((source/'input_fingerprints_after.json').read_text()) if x['path']==str(TIFF))
    stat_before = TIFF.stat()
    assert stat_before.st_size==source_fingerprint['bytes'] and stat_before.st_mtime_ns==source_fingerprint['mtime_ns']
    page_hashes = {x['frame']:x['pixel_sha256'] for x in json.loads((source/'source_page_manifest.json').read_text())}
    out = source/f'raw_error_overlays_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    out.mkdir(exist_ok=False)
    (Path(__file__).parent/'latest_error_overlays_output.txt').write_text(str(out)+'\n')
    manifest=[]
    with tifffile.TiffFile(TIFF) as tf:
        assert tf.series[0].shape==(2500,684,428) and tf.series[0].axes=='TYX'
        for frame in frames:
            raw = tf.pages[frame].asarray()
            assert raw.shape==(684,428) and raw.dtype==np.uint16
            assert hashlib.sha256(raw.tobytes()).hexdigest()==page_hashes[frame]
            group = table[table.frame.eq(frame)]
            manifest.append(render(raw,group,vertices,preview,frame,out))
    after = [fingerprint(p) for p in inputs]
    assert before==after
    assert TIFF.stat().st_size==stat_before.st_size and TIFF.stat().st_mtime_ns==stat_before.st_mtime_ns
    fits.to_csv(out/'selected_original_ROI_fit_rows.csv',index=False)
    table.to_csv(out/'selected_overlay_rows_with_original_centers.csv',index=False)
    detections.to_csv(out/'selected_all_FOV_TrackMate_rows.csv',index=False)
    pd.DataFrame(manifest).to_csv(out/'frame_manifest.csv',index=False)
    dump(out/'config.json',dict(source=str(source),frames=frames,layout='raw image and cached current 7x7 error overlay',
        crop_xyxy_px=[82,232,347,430],display_scale=3,contrast_ADU=preview['resolved_display_range_adu'],
        pixel_size_nm=117,frame_interval_s=.02,background_method=rules['fit_rules']['noise_model'],
        fit_rules=rules['fit_rules'],error_extent='uncapped +/-1 full-Hessian per-axis model SE',
        calibration_status='manufacturer_nominal',no_refitting=True,no_detection=True,no_TARDIS=True))
    dump(out/'source_fingerprints.json',dict(cache_ROI_MAT=before,
        TIFF_whole_file_fingerprint_from_completed_analysis=source_fingerprint,
        TIFF_checks_this_render='size and mtime rechecked; all three source page pixel SHA256 verified'))
    dump(out/'validation.json',dict(original_values_and_QC_unchanged=True,source_files_unchanged=True,
        ROI_MAT_454_vertices_exactly_equal=True,original_frames_times_and_candidates_preserved=True,
        source_page_hashes_match=True,raw_panel_pixels_exact=True,counts_match_cache=True,
        no_refitting=True,no_detection=True,no_error_capping=True,coverage_validated=False))
    snap = out/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),Path(draw_lib.__file__),
        Path(__file__).parents[1]/'trackmate_python_bridge/run_preview.py',Path(roi_lib.__file__)]:
        shutil.copyfile(p,snap/p.name)
    dump(out/'versions.json',dict(python=sys.version,packages={k:importlib.metadata.version(k)
        for k in ['numpy','pandas','Pillow','tifffile','PyYAML']}))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),
        '--source',str(source),'--frames',*[str(x) for x in frames]])+'\n')
    (out/'REPORT.txt').write_text('Original frames 200,1200,2000 = 4,24,40 s.\n'
        'Two panels: raw C2 single frame, same image plus current cached 7x7 annular-background fit centers and per-axis errors.\n'
        'Green=unflagged; magenta=overlap review; orange X=unavailable error at original TrackMate center; yellow=cell ROI.\n'
        'The orange X is a position marker, not an error bar. No finite error is invented for excluded/bound fits.\n'
        'No refitting or detection. No box5 comparison, trajectories, TARDIS, masking or coordinate transform.\n'
        'Errors are local full-Hessian model SE under nominal calibration, not native Picasso CRLB or verified actual accuracy.\n')
    dump(out/'COMPLETED.json',dict(all_checks_passed=True,output=str(out),frames=frames,
        no_refitting=True,manifest=manifest,uncertainty_coverage_validated=False))
    print(json.dumps(dict(output=str(out),manifest=manifest),indent=2),flush=True)


if __name__=='__main__':
    main()
