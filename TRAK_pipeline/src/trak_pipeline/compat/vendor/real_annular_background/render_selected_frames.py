"""Four-panel selected-frame QC using latest cached annular 7x7 fits.

Reuse TrackMate candidates and the exact saved background support; fit 5x5
only for requested frames. No new detection, temporal analysis or masking.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import importlib.metadata
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
import yaml

import run_real as core
from run_full import decorate
from run_preview import draw_lib


def render(raw, candidates, box7, box5, vertices, preview, frame, out):
    c = dict(preview, display_scale=3)
    lo, hi = c['resolved_display_range_adu']
    gray = Image.fromarray(draw_lib.display_gray(raw,lo,hi,3)).convert('RGBA')
    points = candidates[candidates.roi_inside.eq(1)][['x_px','y_px']].to_numpy()
    detection = Image.alpha_composite(gray,draw_lib.overlay_layer(raw.shape,points,vertices,c))
    panels = [gray.convert('RGB'),detection.convert('RGB')]
    counts = []
    colors = dict(clean='#35E6A0',review='#F28DE9',unavailable='#FFAB40')
    for table in [box7,box5]:
        layer = draw_lib.overlay_layer(raw.shape,np.empty((0,2)),vertices,c)
        draw = ImageDraw.Draw(layer)
        valid = 0
        for row in table.itertuples():
            if row.histogram_eligible:
                x,y = draw_lib.display_center([row.fit_x_px,row.fit_y_px],3)
                dx,dy = row.x_se_nm/117*3,row.y_se_nm/117*3
                assert np.isfinite([x,y,dx,dy]).all() and min(dx,dy)>0
                color = colors['clean' if row.qc_unflagged_model_fit else 'review']
                draw.line([(x-dx,y),(x+dx,y)],fill=color,width=1)
                draw.line([(x,y-dy),(x,y+dy)],fill=color,width=1)
                draw.ellipse((x-1.3,y-1.3,x+1.3,y+1.3),fill=color)
                valid += 1
            else:
                x,y = draw_lib.display_center([row.x_px,row.y_px],3)
                draw.line([(x-3,y-3),(x+3,y+3)],fill=colors['unavailable'],width=1)
                draw.line([(x-3,y+3),(x+3,y-3)],fill=colors['unavailable'],width=1)
        assert valid == int(table.histogram_eligible.sum())
        counts.append(valid)
        panels.append(Image.alpha_composite(gray,layer).convert('RGB'))
    crop = tuple(3*v for v in (82,232,347,430))
    w,h = crop[2]-crop[0],crop[3]-crop[1]
    header,margin,gap = 104,14,14
    width = 4*w+2*margin+3*gap
    image = Image.new('RGB',(width,h+220),'#101317')
    draw = ImageDraw.Draw(image)
    big = ImageFont.truetype(c['font_path'],27)
    small = ImageFont.truetype(c['font_path'],23)
    detail = ImageFont.truetype(c['font_path'],19)
    draw.text((margin,12),f'T1 C2 | Original frame {frame:04d} | {frame*.02:.2f} s | TrackMate 0.28 um | Latest annular-background method',font=big,fill='white')
    labels = ['Raw single frame',f'TrackMate | {len(points)} ROI candidates',
        f'7x7 | {counts[0]} available errors',f'5x5 | {counts[1]} available errors']
    for j,(panel,label) in enumerate(zip(panels,labels)):
        x = margin+j*(w+gap)
        draw.text((x,60),label,font=small,fill='white')
        image.paste(panel.crop(crop),(x,header))
    foot = header+h+12
    lines = [
        'Green: unflagged fit | Magenta: overlap review | Orange X: unavailable / excluded error at original TrackMate center | Yellow: cell ROI',
        'Crosses: uncapped +/-1 full-Hessian X/Y model SE. Nominal calibration; NOT native Picasso CRLB or validated accuracy.',
        f'Same candidates, annular background and sigma bounds [0.5,3.5] px. Fixed {lo:g}-{hi:g} ADU. C2 grid; cell crop for display only.']
    for j,text in enumerate(lines):
        draw.text((margin,foot+j*30),text,font=detail,fill='#DCE3EB')
    meta = PngImagePlugin.PngInfo()
    info = dict(original_frame=frame,time_s=frame*.02,ROI_candidates=len(points),
        box7_available=counts[0],box5_available=counts[1],background_method='latest_annular_mean_with_uncertainty')
    for k,v in info.items():
        meta.add_text(k,str(v))
    path = out/f'frame_{frame:04d}_raw_TM_box7_box5.png'
    image.save(path,pnginfo=meta,compress_level=5)
    with Image.open(path) as read:
        for k,v in info.items():
            assert read.info[k]==str(v)
        np.testing.assert_array_equal(np.asarray(read.crop((margin,header,margin+w,header+h))),
            np.asarray(gray.convert('RGB').crop(crop)))
    return dict(**info,image=str(path),source_pixel_sha256=hashlib.sha256(raw.tobytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--frames',nargs='+',type=int,default=[200,1200,2000])
    args = ap.parse_args()
    source = args.source.resolve()
    frames = args.frames
    assert len(frames)==len(set(frames)) and all(0<=f<2500 for f in frames)
    rules = json.loads((source/'rules_before_run.json').read_text())
    c = rules['fit_rules']; camera = rules['camera']
    assert c['box_px']==7 and c['pixel_size_nm']==117 and c['frame_interval_s']==.02
    preview = yaml.safe_load((Path(c['source_preview'])/'resolved_config.yaml').read_text())
    inputs = [source/name for name in ['all_detections.csv','all_ROI_target_fits.csv',
        'annular_background_all_ROI_targets.csv','retained_annular_pixels.npz','roi_vertices.csv']]
    inputs += [Path(preview[name]) for name in ['input_tiff','roi_file','mat_file']]
    before = [core.fingerprint(p) for p in inputs]
    assert before[-3]['sha256']==preview['expected_tiff_sha256']
    assert before[-2]['sha256']==preview['expected_roi_sha256']
    detections = pd.read_csv(inputs[0],float_precision='round_trip')
    all7 = pd.read_csv(inputs[1],low_memory=False,float_precision='round_trip')
    allbg = pd.read_csv(inputs[2],float_precision='round_trip')
    masks = np.load(inputs[3])
    vertices = core.roi_lib.load_imagej_roi(preview['roi_file'])
    np.testing.assert_array_equal(vertices,pd.read_csv(inputs[4]).to_numpy())
    mat_vertices,_ = core.read_mat_vertices(preview['mat_file'],preview['mat_record'])
    np.testing.assert_array_equal(vertices,mat_vertices)
    np.testing.assert_array_equal(all7.candidate_index,allbg.candidate_index)
    np.testing.assert_array_equal(all7.candidate_index,masks['candidate_index'])
    detections = detections[detections.frame.isin(frames)].copy()
    fit7 = all7[all7.frame.isin(frames)].copy()
    targets = detections[detections.roi_inside.eq(1)].copy()
    np.testing.assert_array_equal(fit7.candidate_index,targets.candidate_index)
    bg = allbg[allbg.frame.isin(frames)].copy()
    selected_indices = np.flatnonzero(all7.frame.isin(frames))
    selected_masks = {int(all7.candidate_index.iloc[j]):
        masks['xy_pixels'][masks['offsets'][j]:masks['offsets'][j+1]].copy() for j in selected_indices}
    out = source/f'selected_frame_box_comparison_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    out.mkdir(exist_ok=False)
    (Path(__file__).parent/'latest_selected_frames_output.txt').write_text(str(out)+'\n')
    core.dump(out/'config.json',dict(source=str(source),frames=frames,box7='cached, unchanged',
        box5='new fits at identical TrackMate candidates',fit_rules_box7=c,
        fit_rules_box5=dict(c,box_px=5),camera=camera,contrast_adu=preview['resolved_display_range_adu'],
        identical_width_bounds_px=c['width_bounds_px'],uncertainty='local full-Hessian SE, not native Picasso CRLB',
        native_C2_coordinates=True,errorbar_extent_not_capped=True,tracking=False,tardis=False))
    page_hashes = {x['frame']:x['pixel_sha256'] for x in json.loads((source/'source_page_manifest.json').read_text())}
    new_rows = []
    raw_frames = {}
    with tifffile.TiffFile(preview['input_tiff']) as tf, concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert tf.series[0].shape==(2500,684,428) and tf.series[0].axes=='TYX'
        for frame in frames:
            raw = tf.pages[frame].asarray()
            assert hashlib.sha256(raw.tobytes()).hexdigest()==page_hashes[frame]
            raw_frames[frame] = raw
            movie = (raw.astype(float)-camera['baseline_adu'])*camera['sensitivity_e_per_adu']/camera['em_gain']
            candidates = targets[targets.frame.eq(frame)]
            background = bg[bg.frame.eq(frame)].set_index('candidate_index')
            jobs = []
            for row in candidates.itertuples():
                i = int(row.candidate_index); x,y = row.x_px,row.y_px
                ax,ay = int(np.floor(x+.5)),int(np.floor(y+.5))
                patch7 = raw[ay-3:ay+4,ax-3:ax+4]
                patch5 = raw[ay-2:ay+3,ax-2:ax+3]
                np.testing.assert_array_equal(patch5,patch7[1:6,1:6])
                pixels = selected_masks[i]
                values = movie[pixels[:,1],pixels[:,0]]
                b = background.loc[i]
                np.testing.assert_allclose(values.mean(),b.mean_e,rtol=1e-12)
                jobs.append(pool.submit(core.fit_one,(i,int(row.spot_id),frame,x,y),
                    raw,movie,b,pixels,dict(c,box_px=5)))
            new_rows.extend(f.result() for f in jobs)
            print(f'Frame {frame}: retained {len(candidates)} identical ROI candidates; 5x5 fits completed',flush=True)
    fit5 = decorate(pd.DataFrame(new_rows),vertices,dict(c,box_px=5))
    original_xy = targets.set_index('candidate_index')[['x_px','y_px','source_detection_uid']]
    fit5 = fit5.join(original_xy,on='candidate_index')
    plot7 = fit7.join(original_xy,on='candidate_index')
    # Rendering additions never change cached 7x7 fit/QC fields.
    pd.testing.assert_frame_equal(plot7[fit7.columns],fit7)
    fit7.to_csv(out/'selected_ROI_fits_box7.csv',index=False)
    fit5.to_csv(out/'selected_ROI_fits_box5.csv',index=False)
    detections.to_csv(out/'all_selected_frame_TrackMate_candidates.csv',index=False)
    bg.to_csv(out/'selected_cached_annular_background.csv',index=False)
    manifest = []
    for frame in frames:
        manifest.append(render(raw_frames[frame],detections[detections.frame.eq(frame)],
            plot7[plot7.frame.eq(frame)],fit5[fit5.frame.eq(frame)],vertices,preview,frame,out))
    pd.DataFrame(manifest).to_csv(out/'frame_manifest.csv',index=False)
    statuses = {str(frame):dict(ROI_candidates=int(targets.frame.eq(frame).sum()),
        box7=fit7[fit7.frame.eq(frame)].audit_status.value_counts().to_dict(),
        box5=fit5[fit5.frame.eq(frame)].audit_status.value_counts().to_dict()) for frame in frames}
    core.dump(out/'summary.json',statuses)
    after = [core.fingerprint(p) for p in inputs]
    assert before==after
    core.dump(out/'input_fingerprints.json',before)
    core.dump(out/'validation.json',dict(source_fingerprints_unchanged=True,
        no_new_detection=True,no_7x7_refit=True,exact_same_candidates=True,
        identical_saved_background_support_and_values=True,ROI_MAT_vertices_equal=True,
        raw_panels_pixel_exact=True,source_TIFF_pages_verified=True,
        box5_is_center_crop_of_box7=True,all_candidates_preserved=True,errors_not_capped=True,
        uncertainty_coverage_validated=False))
    snap = out/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),Path(core.__file__),Path(core.base.__file__),
        Path(__file__).with_name('run_full.py'),core.ROOT/'localization_overlays/render_frames.py',
        core.ROOT/'trackmate_python_bridge/run_preview.py',
        core.ROOT/'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py']:
        shutil.copyfile(p,snap/p.name)
    core.dump(out/'versions.json',dict(python=sys.version,packages={k:importlib.metadata.version(k)
        for k in ['numpy','scipy','pandas','Pillow','tifffile']},fitting_backend='independent_annular_constraint_MLE'))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),
        '--source',str(source),'--frames',*[str(x) for x in frames]])+'\n')
    (out/'REPORT.txt').write_text(
        'Requested frames 200, 1200, 2000 (4,24,40 s), latest annular-background method.\n'
        'Same TrackMate 0.28 um candidates: raw, cyan position circles, cached 7x7, new 5x5.\n'
        'Both boxes use the same annular mean, uncertainty constraint, empirical variance and width bounds [0.5,3.5] px.\n'
        'Green=unflagged; magenta=overlap review; orange X=unavailable/excluded error at original TrackMate center.\n'
        'Crosses are uncapped +/-1 full-Hessian per-axis model standard errors, not native Picasso CRLB or calibrated accuracy.\n'
        'The supplied screenshot used an earlier native Picasso model; only its four-panel display style is reused.\n'
        'No candidate deletion, new detection, trajectories, TARDIS, masking or coordinate transform.\n'
        'Identical 406-554 ADU display; native pixels, no frame averaging, crop is display-only.\n')
    core.dump(out/'COMPLETED.json',dict(all_checks_passed=True,output=str(out),frames=frames,
        new_fits=len(fit5),cached_fits=len(fit7),uncertainty_coverage_validated=False))
    print(json.dumps(dict(output=str(out),summary=statuses),indent=2),flush=True)


if __name__=='__main__':
    main()
