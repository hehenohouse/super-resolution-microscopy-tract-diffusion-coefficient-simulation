#!/usr/bin/env python3
"""Pool cached, QC-valid ROI localization centers across all original frames.

Only viewing artifacts are generated. No detection, fitting, classification,
tracking, mask processing or TARDIS calculations are run.
"""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont,PngImagePlugin
import tifffile

from point_stack_core import (display_centers,marker_alpha,roi_alpha,
                              composite,combined_layer,ON,OFF,ROI_COLOR)
from render_point_stack import fingerprint,dump,HERE,PROJECT,pixel_hash


def count_grid(xy,crop,scale):
    """Exact counts in display bins; no smoothing or precision weighting."""
    x0,y0,x1,y1=crop
    edges_x=np.linspace(x0-.5,x1-.5,(x1-x0)*scale+1)
    edges_y=np.linspace(y0-.5,y1-.5,(y1-y0)*scale+1)
    counts=np.histogram2d(xy[:,1],xy[:,0],bins=(edges_y,edges_x))[0].astype(np.uint32)
    assert int(counts.sum())==len(xy)
    return counts


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True,help='Completed six-panel viewing run')
    args=ap.parse_args();run=args.run.resolve()
    assert json.loads((run/'COMPLETED.json').read_text())['delivery_checks_passed']
    parent_params=json.loads((run/'parameters.json').read_text())
    source_run=Path(parent_params['source_run'])
    base=Path(parent_params['C2_file']).parent
    out=base/'analysis_results/mito_point_projections'/(
        'T1_C2_W5min32_all_timepoints_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False);(out/'source_snapshot').mkdir()
    expected={r['path']:r['sha256'] for r in json.loads((run/'output_fingerprints.json').read_text())}
    expected.update({r['path']:r['sha256'] for r in json.loads((run/'input_fingerprints_before.json').read_text())})
    paths=[run/'point_source.csv',run/'parameters.json',source_run/'roi_vertices.csv',source_run/'per_frame_counts.csv']
    before=[fingerprint(p) for p in paths]
    for row in before:assert row['sha256']==expected[row['path']],row['path']
    cfg=dict(source_point_stack=str(run),source_run=str(source_run),frames=[0,2499],
        frame_interval_s=.02,pixel_size_nm=117,crop_xyxy_px=[82,232,347,430],
        display_scale=4,dot_radius_native_px=.20,
        selection='Existing QC-pass, refined center inside shared cell ROI, finite positive X/Y SE; no additional SE threshold',
        temporal_projection='All localization observations pooled; no deduplication or tracking',
        classification='Existing W5/min32 pointwise k=1 labels, unmodified',
        registration_status='exploratory_identity_registration_unverified',
        rendering='Floating-center fixed-size dots, maximum antialiased coverage within group; not a density or probability image',
        combined_overlap='Commutative coverage-weighted On/Not-on color mixture',
        scientific_count_grids='4x raster counts, 29.25 nm bin width; no smoothing; distinct from dot-viewing image',
        font_path=parent_params['font_path'],background='black',ROI='thin gray outline',
        color_On=ON.astype(int).tolist(),color_Not_on=OFF.astype(int).tolist(),
        precision_status='Current nominal-calibration model estimates; dot size does not represent precision',
        refitting=False,reclassification=False,TARDIS=False)
    table=pd.read_csv(run/'point_source.csv',float_precision='round_trip')
    assert len(table)==115013 and table.candidate_index.is_unique
    assert table.qc_unflagged_model_fit.all() and table.fitted_center_inside_ROI.all() and table.spatial_selected.all()
    assert table.frame.between(0,2499).all() and table.frame.nunique()==2500
    assert table.spatial_registration_status.eq(cfg['registration_status']).all()
    valid=np.isfinite(table[['x_se_nm','y_se_nm','fit_x_px','fit_y_px']]).all(axis=1)
    valid &= table[['x_se_nm','y_se_nm']].gt(0).all(axis=1)
    assert valid.all(),'Completed cache contains unexpected invalid precision; stop rather than silently change membership'
    on=table[table.tardis_group=='On'];off=table[table.tardis_group=='Not-on']
    assert len(on)==57615 and len(off)==57398 and len(on)+len(off)==len(table)
    assert set(on.candidate_index).isdisjoint(set(off.candidate_index))
    counts=pd.read_csv(source_run/'per_frame_counts.csv')
    np.testing.assert_array_equal(counts.frame,np.arange(2500))
    for group in ['On','Not-on']:
        np.testing.assert_array_equal(table[table.tardis_group==group].groupby('frame').size().reindex(range(2500),fill_value=0),counts[group])
    assert int(counts.QC_excluded.sum())==12046
    vertices=pd.read_csv(source_run/'roi_vertices.csv').to_numpy(float)
    crop=cfg['crop_xyxy_px'];scale=cfg['display_scale'];w=(crop[2]-crop[0])*scale;h=(crop[3]-crop[1])*scale
    xy_on=on[['fit_x_px','fit_y_px']].to_numpy();xy_off=off[['fit_x_px','fit_y_px']].to_numpy()
    pos_on=display_centers(xy_on,crop,scale);pos_off=display_centers(xy_off,crop,scale)
    for pos in [pos_on,pos_off]:
        assert (pos>=0).all() and (pos[:,0]<w).all() and (pos[:,1]<h).all()
    alpha_on=marker_alpha((h,w),pos_on,cfg['dot_radius_native_px']*scale)
    alpha_off=marker_alpha((h,w),pos_off,cfg['dot_radius_native_px']*scale)
    base_image=composite(np.zeros((h,w,3),np.uint8),roi_alpha(vertices,crop,scale),ROI_COLOR)
    panels=[composite(base_image,alpha_on,ON),composite(base_image,alpha_off,OFF),
            combined_layer(base_image,alpha_on,alpha_off)]
    # Order-independent overlap and exact group-union count-map checks.
    np.testing.assert_array_equal(alpha_on,marker_alpha((h,w),pos_on[::-1],cfg['dot_radius_native_px']*scale))
    maps_on=count_grid(xy_on,crop,scale);maps_off=count_grid(xy_off,crop,scale)
    maps_all=count_grid(table[['fit_x_px','fit_y_px']].to_numpy(),crop,scale)
    np.testing.assert_array_equal(maps_all,maps_on+maps_off)
    np.savez_compressed(out/'localization_count_maps.npz',On=maps_on,Not_on=maps_off,all_valid=maps_all,
                        native_crop_xyxy_px=np.array(crop),nm_per_bin=np.array(117/scale))
    font=lambda n:ImageFont.truetype(cfg['font_path'],n)
    names=['on_all_timepoints','not_on_all_timepoints','all_valid_all_timepoints']
    titles=[f'On | {len(on):,} localizations',f'Not-on | {len(off):,} localizations',f'All valid: On + Not-on | {len(table):,} localizations']
    images=[]
    for i,(panel,name,title) in enumerate(zip(panels,names,titles)):
        image=Image.new('RGB',(w,h+128),'black');image.paste(Image.fromarray(panel),(0,80))
        draw=ImageDraw.Draw(image)
        draw.text((12,8),'TDKd3 T1 | ALL original frames 0-2499 pooled | W5/min32',font=font(22),fill='white')
        color=tuple(ON.astype(int)) if i==0 else tuple(OFF.astype(int)) if i==1 else 'white'
        draw.text((12,43),title,font=font(25),fill=color)
        # 2 um scale bar, in the same native-coordinate mapping as the points.
        bx,by=display_centers([[96,424]],crop,scale)[0]
        draw.line([(bx,80+by),(bx+2000/117*scale,80+by)],fill='white',width=3)
        draw.text((bx,80+by-23),'2 um',font=font(17),fill='white')
        draw.text((12,h+88),'Dots mark localization centers, not their errors; repeat observations retained.',font=font(18),fill='#DDDDDD')
        draw.text((12,h+111),'Cell ROI only | C1/C2 identity registration unverified | QC/precision-valid cache',font=font(17),fill='#BBBBBB')
        metadata=PngImagePlugin.PngInfo();metadata.add_text('parameters',json.dumps(cfg));metadata.add_text('localizations',str([len(on),len(off),len(table)][i]))
        image.save(out/f'{name}.png',pnginfo=metadata,compress_level=6)
        tifffile.imwrite(out/f'{name}.tif',np.asarray(image),photometric='rgb',compression='deflate',
            metadata={'axes':'YXS','Description':'Static all-timepoint viewing composite; not a movie or scientific intensity image'})
        with Image.open(out/f'{name}.png') as check:np.testing.assert_array_equal(np.asarray(check),np.asarray(image))
        np.testing.assert_array_equal(tifffile.imread(out/f'{name}.tif'),np.asarray(image))
        images.append(image)
    overview=Image.new('RGB',(3*w+64,h+128),'black')
    for i,image in enumerate(images):overview.paste(image,(i*(w+32),0))
    overview.save(out/'all_timepoints_triptych.png',compress_level=6)
    shutil.copyfile(run/'point_source.csv',out/'localization_source.csv')
    assert fingerprint(out/'localization_source.csv')['sha256']==before[0]['sha256']
    for script in ['render_time_projection.py','point_stack_core.py','render_point_stack.py']:
        shutil.copyfile(HERE/script,out/'source_snapshot'/script)
    shutil.copyfile(PROJECT/'picasso_pilot/requirements.lock.txt',out/'requirements.lock.txt')
    dump(out/'parameters.json',cfg)
    dump(out/'software_versions.json',dict(python=sys.version,packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','Pillow','tifffile']}))
    command=shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--run',str(run)])
    (out/'command.txt').write_text(command+'\n')
    summary=dict(On=len(on),Not_on=len(off),all_valid=len(table),frames=2500,original_frames=[0,2499],
        original_ROI_candidates=127059,original_QC_excluded=12046,additional_precision_exclusions=0,
        all_included_have_finite_positive_X_Y_SE=True,third_image_is_exact_union=True,
        counts_are_observations_not_unique_molecules=True)
    dump(out/'selection_summary.json',summary)
    after=[fingerprint(p) for p in paths];assert before==after
    dump(out/'input_fingerprints_before.json',before);dump(out/'input_fingerprints_after.json',after)
    dump(out/'validation.json',dict(source_fingerprints_unchanged=True,all_points_in_common_crop=True,
        all2500_original_frames_included=True,per_frame_counts_match_cache=True,
        count_maps_conserve_all_localizations=True,combined_count_map_equals_group_sum=True,
        candidate_groups_disjoint=True,dot_projection_invariant_to_observation_order=True,
        all_PNG_TIFF_pixels_identical=True,no_refitting_reclassification_or_TARDIS=True))
    (out/'README.txt').write_text(
        'Three static all-timepoint localization projections: On, Not-on, and their exact union.\n'
        'Pool all original frames0-2499 (0-49.98s); retain repeat observations, not unique molecules.\n'
        'Same current QC, refined ROI-center selection, W5/min32 k1 labels. X/Y SE finite and positive, no further SE threshold.\n'
        'On57615, Not-on57398, union115013. Original127059 ROI candidates include12046 QC exclusions; no QC-excluded rows restored.\n'
        'Black background; common gray ROI; native crop x[82,347),y[232,430),x-right/y-down.\n'
        'Display4x, fixed dot radius0.20 native pixel, subpixel centers. Marker size is not localization precision.\n'
        'On green, Not-on purple, overlapping marker colors mix without draw-order preference.\n'
        'These PNG/RGB TIFF files are display projections with headers and scalebars, not density/precision/molecule-count maps.\n'
        'Count maps NPZ separately store exact unsmoothed4x-grid counts (29.25nm/bin), with group sum and total conservation verified.\n'
        'Current nominal-calibration model uncertainties and unverified C1/C2 identity registration are retained.\n'
        'Sources unchanged; no image or mask processing, refitting, reclassification, tracking or TARDIS.\n')
    entry=(f'\n[2026-10-16] TDKd3：全部2500帧定位叠加为三张静态图\n输出：{out}\n'
        'On57615，Not-on57398，合并115013；沿用当前QC和共同ROI，X/Y精度均有效；12046个原QC排除点不恢复。\n'
        '全部原始帧0–2499保留并汇集，不去重，不代表独立分子数；原坐标、误差和W5/min32分类不变。\n'
        '三张PNG/RGB TIFF及并排预览：黑底小点、灰色ROI、绿色On、紫色Not-on，合并重叠混色；坐标裁切一致，2um标尺。\n'
        '另存精确计数NPZ和原点表，核验总数与逐帧缓存一致，组间不重叠，合并计数等于两组之和；PNG/TIFF像素一致，源指纹不变。\n'
        '只做静态查看图，不重拟合、重分类或运行TARDIS；恒等配准仍未经独立验证。\n')
    for log in [PROJECT/'real_annular_background/LOG.txt',base/'analysis_results/annular_background_refinement/ANALYSIS_LOG.txt']:
        with log.open('a') as f:f.write(entry)
    (out/'log_entry.txt').write_text(entry)
    dump(out/'COMPLETED.json',dict(complete=True,counts=summary,output=str(out),visual_review_pending=True))
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']])
    (HERE/'latest_time_projection_output.txt').write_text(str(out)+'\n')
    print(json.dumps(dict(output=str(out),counts=summary),indent=2))


if __name__=='__main__':main()
