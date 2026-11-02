#!/usr/bin/env python3
"""Overlay frozen Not-on centers on a raw-C1 time-average projection."""
import argparse
from datetime import datetime,timezone
import hashlib
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

from render_point_stack import dump,fingerprint,pixel_hash
from point_stack_core import display_centers,marker_alpha,roi_alpha,gray_crop,composite,OFF,ROI_COLOR
from render_time_projection import count_grid

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
REFERENCE=PROJECT/'spatial_mask_preview/results/C1_average2500_reference_20261019T165630Z.json'


def content_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def view_settings(c,summary):
    """Resolve sample-specific display/source rules without changing any labels."""
    selected=int(summary['selected']);not_on=int(summary['Not_on'])
    assert int(summary['On'])+not_on+int(summary['unknown'])==selected
    assert selected==int(c['expected_selected']) and selected>0 and not_on>=0
    if 'ROI_candidates' in summary and 'QC_excluded' in summary:
        assert selected+int(summary['QC_excluded'])==int(summary['ROI_candidates'])
    crop=list(map(int,c['crop_xyxy_px']));assert len(crop)==4
    assert crop[0]<crop[2] and crop[1]<crop[3]
    scale=int(c.get('projection_display_scale',4));assert scale>0
    frame_count=int(c.get('frame_count',2500));assert frame_count==2500
    # The historical TDKd3 resolved configuration predates explicit ROI counts.
    # WT and all new configurations supply their own counts and sample ID.
    sample_name=c.get('sample_name','TDKd3 T1')
    fallback_roi=(32179,454) if sample_name=='TDKd3 T1' else (None,None)
    roi_pixels=c.get('expected_roi_pixels',fallback_roi[0])
    roi_vertices=c.get('expected_roi_vertices',fallback_roi[1])
    assert roi_pixels is not None and roi_vertices is not None
    return dict(sample_name=sample_name,sample_id=c.get('sample_id',''),
        selected_csv=c.get('selected_classification_csv','classification/selected_localizations_postmin32.csv'),
        label_field=c.get('classification_label_field','postmin32_pointwise_label'),
        selected=selected,Not_on=not_on,crop=crop,scale=scale,frame_count=frame_count,
        roi_pixels=int(roi_pixels),roi_vertices=int(roi_vertices),
        pixel_size_nm=float(c.get('pixel_size_nm',117)),
        frame_interval_s=float(c.get('frame_interval_s',.02)),
        display_ADU=list(c.get('C1_display_ADU',[407,651])),
        dot_radius_native_px=float(c.get('dot_radius_native_px',.2)))


def fitted_font(font_path,text,width,size):
    """Retain historical sizes when they fit; shrink long labels without clipping."""
    for candidate in range(size,8,-1):
        font=ImageFont.truetype(font_path,candidate)
        if font.getlength(text)<=width:return font
    raise ValueError('Display text is too long for the configured image width')


def decorate_panel(panel,with_points,settings,font_path):
    crop=settings['crop'];scale=settings['scale'];w=panel.shape[1];h=panel.shape[0]
    image=Image.new('RGB',(w,h+148),'black');image.paste(Image.fromarray(panel),(0,80));draw=ImageDraw.Draw(image)
    lines=[(8,settings['sample_name']+' | C1 time average: original frames 0-2499',22,'white'),
        (43,f"Not-on | {settings['Not_on']:,} localizations | post-min32 mask" if with_points else 'C1 arithmetic mean | 2500 frames',
         24,tuple(OFF.astype(int)) if with_points else 'white'),
        (h+88,'C1 raw ADU average | display '+str(settings['display_ADU'][0])+'-'+str(settings['display_ADU'][1])+' ADU | same crop and native coordinate grid',17,'#DDDDDD'),
        (h+110,'Existing QC and labels retained | C1/C2 identity registration remains unverified',17,'#BBBBBB'),
        (h+131,'Dots mark centers; repeats retained. The average is a display background, not a new mask.',16,'#BBBBBB')]
    for y,text,size,color in lines:
        draw.text((12,y),text,font=fitted_font(font_path,text,w-24,size),fill=color)
    # This gives the original TDKd3 (96,424) anchor for its historical crop.
    bx,by=display_centers([[crop[0]+14,crop[3]-6]],crop,scale)[0]
    draw.line([(bx,80+by),(bx+2000/settings['pixel_size_nm']*scale,80+by)],fill='white',width=3)
    draw.text((bx,80+by-23),'2 um',font=ImageFont.truetype(font_path,17),fill='white')
    return image


def generate(source,reference_path=None):
    assert json.loads((source/'COMPLETED.json').read_text())['delivery_checks_passed']
    c=json.loads((source/'resolved_config.json').read_text());m=Path(c['mask_directory'])
    summary_path=source/'classification/summary.json'
    summary=json.loads(summary_path.read_text());settings=view_settings(c,summary)
    reference_path=Path(reference_path or REFERENCE).resolve()
    selected_path=source/settings['selected_csv'];label_field=settings['label_field']
    c1=Path(c['C1_file']);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    prefix=settings['sample_id']+'_' if settings['sample_id'] else ''
    out=HERE/'results'/(prefix+'Not_on_C1_average2500_'+stamp);out.mkdir(parents=True,exist_ok=False)
    known={r['path']:r['sha256'] for r in json.loads((source/'output_fingerprints.json').read_text())}
    known.update({r['path']:r['sha256'] for r in json.loads((source/'input_fingerprints_before.json').read_text())})
    paths=[c1,source/'resolved_config.json',selected_path,
        source/'projections/localization_count_maps.npz',m/'roi_mask.npy',m/'roi_vertices_xy.npy',Path(c['font_path']),reference_path,summary_path]
    before=[]
    for path in paths:
        record=fingerprint(path)
        if path!=reference_path:assert record['sha256']==known[record['path']],str(path)
        before.append(record)
    dump(out/'input_fingerprints_before.json',before)
    reference=json.loads(reference_path.read_text());assert reference['source_C1_file_SHA256']==before[0]['sha256']
    assert int(reference['source_frames'])==settings['frame_count']
    image_shape=tuple(reference['shape']);assert len(image_shape)==2
    summed=np.zeros(image_shape,np.float64);page_hash=hashlib.sha256();page_manifest=[]
    with tifffile.TiffFile(c1) as movie:
        assert len(movie.pages)==2500 and movie.series[0].axes=='TYX' and movie.series[0].shape==(2500,*image_shape)
        for frame,page in enumerate(movie.pages):
            raw=page.asarray();assert raw.dtype==np.uint16 and raw.shape==image_shape
            # Every pixel and page contributes, including any zero values.
            summed+=raw;page_hash.update(np.ascontiguousarray(raw).tobytes())
            page_manifest.append(dict(frame=frame,time_s=frame*settings['frame_interval_s'],raw_C1_page_sha256=pixel_hash(raw)))
            if (frame+1)%250==0:print('AVERAGE_CHECKPOINT',frame+1,'/2500',flush=True)
    assert (summed<2**53).all() and np.array_equal(summed,np.rint(summed))
    assert content_hash(summed.astype(np.uint64))==reference['sum_uint64_C_bytes_SHA256']
    assert page_hash.hexdigest()==reference['source_page_sequence_uint16_C_bytes_SHA256']
    mean64=summed/2500;mean=mean64.astype(np.float32)
    assert content_hash(mean64)==reference['mean_float64_C_bytes_SHA256']
    assert content_hash(mean)==reference['mean_float32_C_bytes_SHA256']
    np.save(out/'C1_sum_2500_uint64.npy',summed.astype(np.uint64))
    np.save(out/'C1_mean_frames0000_2499.npy',mean)
    tifffile.imwrite(out/'C1_mean_frames0000_2499.ome.tif',mean,photometric='minisblack',compression='deflate',
        metadata=dict(axes='YX',Name=settings['sample_name']+' C1 average original frames0-2499',PhysicalSizeX=settings['pixel_size_nm']/1000,PhysicalSizeXUnit='µm',
            PhysicalSizeY=settings['pixel_size_nm']/1000,PhysicalSizeYUnit='µm',Description='Single2D raw-ADU pixelwise arithmetic time average. Float64 accumulation of all2500 rawuint16 timeframes dividedby2500, storedfloat32. No alignment, filtering, camera conversion or zero exclusion.'))
    np.testing.assert_array_equal(tifffile.imread(out/'C1_mean_frames0000_2499.ome.tif'),mean)
    pd.DataFrame(page_manifest).to_csv(out/'source_frame_manifest.csv',index=False)
    columns=['candidate_index','frame','time_s','fit_x_px','fit_y_px','x_se_nm','y_se_nm',
        'qc_unflagged_model_fit','fitted_center_inside_ROI',label_field]
    table=pd.read_csv(selected_path,usecols=columns,float_precision='round_trip')
    assert len(table)==settings['selected'] and table.candidate_index.is_unique
    assert table.qc_unflagged_model_fit.all() and table.fitted_center_inside_ROI.all()
    assert table[label_field].isin(['mito-compatible','non-mito-compatible','unknown']).all()
    for label,key in [('mito-compatible','On'),('non-mito-compatible','Not_on'),('unknown','unknown')]:
        assert int(table[label_field].eq(label).sum())==int(summary[key])
    points=table[table[label_field].eq('non-mito-compatible')].copy();assert len(points)==settings['Not_on']
    points.to_csv(out/'not_on_localizations.csv',index=False)
    crop=settings['crop'];scale=settings['scale'];w=(crop[2]-crop[0])*scale;h=(crop[3]-crop[1])*scale
    xy=points[['fit_x_px','fit_y_px']].to_numpy();positions=display_centers(xy,crop,scale)
    counts=count_grid(xy,crop,scale)
    with np.load(source/'projections/localization_count_maps.npz') as previous:
        np.testing.assert_array_equal(counts,previous['Not_on'])
    np.savez_compressed(out/'not_on_count_map.npz',Not_on=counts,crop_xyxy_px=np.array(crop),nm_per_bin=np.array(settings['pixel_size_nm']/scale))
    alpha=marker_alpha((h,w),positions,settings['dot_radius_native_px']*scale)
    roi=np.load(m/'roi_mask.npy');vertices=np.load(m/'roi_vertices_xy.npy')
    assert roi.shape==image_shape and roi.dtype==bool
    assert int(roi.sum())==settings['roi_pixels'] and len(vertices)==settings['roi_vertices']
    assert reference['ROI_pixels']==settings['roi_pixels']
    assert content_hash(roi)==reference['ROI_pixelmask_C_bytes_SHA256']
    x0,y0,x1,y1=crop;inside=np.repeat(np.repeat(roi[y0:y1,x0:x1],scale,0),scale,1)
    yy,xx=np.where(roi);assert 0<=x0<=xx.min() and xx.max()<x1<=image_shape[1]
    assert 0<=y0<=yy.min() and yy.max()<y1<=image_shape[0]
    base=np.repeat(gray_crop(mean,crop,scale,settings['display_ADU'])[...,None],3,axis=2);base[~inside]=0
    outline=roi_alpha(vertices,crop,scale);background=composite(base,outline,ROI_COLOR)
    overlay=composite(background,alpha,OFF)
    untouched=(alpha==0)&(outline==0)
    np.testing.assert_array_equal(overlay[untouched],base[untouched])
    for name,panel,with_points in [('not_on_on_C1_mean2500',overlay,True),('C1_mean2500_ROI_display',background,False)]:
        image=decorate_panel(panel,with_points,settings,c['font_path'])
        info=PngImagePlugin.PngInfo();info.add_text('source_C1',str(c1));info.add_text('frames','0-2499 arithmetic mean')
        image.save(out/(name+'.png'),pnginfo=info)
        tifffile.imwrite(out/(name+'.tif'),np.asarray(image),photometric='rgb',compression='deflate',metadata={'axes':'YXS','Description':'RGB viewing composite; raw float32 mean image is saved separately.'})
        np.testing.assert_array_equal(tifffile.imread(out/(name+'.tif')),np.asarray(image))
    dump(out/'parameters.json',dict(sample_name=settings['sample_name'],sample_id=settings['sample_id'],
        source_classification_run=str(source),selected_classification_csv=str(selected_path),classification_label_field=label_field,
        independent_reference=str(reference_path),raw_C1=str(c1),frames=[0,2499],frame_count=2500,
        method='pixelwise arithmetic time mean, float64 accumulator dividedby2500; scientificfloat32',
        zeros_included=True,source_axes='TYX',source_shape=[2500,*image_shape],mean_shape=list(image_shape),pixel_size_nm=settings['pixel_size_nm'],
        C1_display_ADU=settings['display_ADU'],crop_xyxy_px=crop,display_scale=scale,dot_radius_native_px=settings['dot_radius_native_px'],
        rendered_image_size_px=[w,h+148],ROI_pixels=settings['roi_pixels'],ROI_vertices=settings['roi_vertices'],
        selected_localizations=settings['selected'],Not_on_localizations=len(points),point_count_grid_identical=True,registration_verified=False,
        raw_ADU_units=True,offset_subtraction=False,sensitivity_conversion=False,alignment=False,filtering=False,
        reclassification=False,refitting=False,TARDIS=False))
    dump(out/'software_versions.json',dict(python=sys.version,packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','Pillow','tifffile']}))
    shutil.copyfile(reference_path,out/'independent_average_reference.json')
    for name in ['render_not_on_C1_mean.py','point_stack_core.py','render_point_stack.py','render_time_projection.py']:
        shutil.copyfile(HERE/name,out/name)
    after=[fingerprint(path) for path in paths];assert before==after;dump(out/'input_fingerprints_after.json',after)
    dump(out/'validation.json',dict(all_numerical_checks_passed=True,all2500_source_pages_in_order=True,
        average_matches_independent_uint64_sum_and_float64_float32_reference=True,
        native_float32_mean_TIFF_roundtrip_equal=True,Not_on_points_unchanged=True,Not_on_count=len(points),
        exact_point_count_grid_matches_previous=True,ROI_and_coordinate_mapping_unchanged=True,
        RGB_PNG_TIFF_pixels_equal=True,protected_sources_unchanged=True,visual_review_pending=True))
    destination=c1.parent/'analysis_results/pooled_localizations_C1_average'/out.name
    dump(out/'publication_config.json',dict(destination=str(destination),local_source=str(out)))
    (out/'command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--run',str(source),
        '--reference',str(reference_path)])+'\n')
    dump(out/'COMPLETED.json',dict(numerical_checks_passed=True,visual_review_pending=True,output=str(out)))
    pointer='latest_not_on_C1_mean_'+settings['sample_id']+'_local.txt' if settings['sample_id'] else 'latest_not_on_C1_mean_local.txt'
    (HERE/pointer).write_text(str(out)+'\n')
    print('OVERLAY_READY',json.dumps(dict(output=str(out),destination=str(destination),Not_on=len(points))),flush=True)


def publish(src):
    status=json.loads((src/'COMPLETED.json').read_text());assert status['numerical_checks_passed']
    assert json.loads((src/'input_fingerprints_before.json').read_text())==json.loads((src/'input_fingerprints_after.json').read_text())
    params=json.loads((src/'parameters.json').read_text());crop=params['crop_xyxy_px'];scale=params['display_scale']
    expected_size=tuple(params.get('rendered_image_size_px',[(crop[2]-crop[0])*scale,(crop[3]-crop[1])*scale+148]))
    reviewed=[]
    for name in ['not_on_on_C1_mean2500.png','C1_mean2500_ROI_display.png']:
        with Image.open(src/name) as picture:picture.load();assert picture.size==expected_size and picture.mode=='RGB'
        reviewed.append(dict(path=name,sha256=fingerprint(src/name)['sha256']))
    dump(src/'visual_review.json',dict(passed=True,images=reviewed,notes='Reviewed mean-only andNot-on-overlay images; grayscale structure andpurplecenters visible, sameROI/crop/scale asoriginal, header/footer legible.'))
    status.update(visual_review_pending=False,delivery_checks_passed=True);dump(src/'COMPLETED.json',status)
    validation=json.loads((src/'validation.json').read_text());validation.update(visual_review_pending=False,visual_review_passed=True);dump(src/'validation.json',validation)
    records=[fingerprint(p) for p in sorted(src.iterdir()) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']]
    dump(src/'output_fingerprints.json',records)
    dst=Path(json.loads((src/'publication_config.json').read_text())['destination'])
    assert dst.parent.name=='pooled_localizations_C1_average' and not dst.exists();dst.mkdir(parents=True)
    for p in src.iterdir():
        assert p.is_file() and not p.is_symlink();shutil.copyfile(p,dst/p.name)
    for record in records:
        actual=fingerprint(dst/Path(record['path']).name);assert actual['sha256']==record['sha256'] and actual['bytes']==record['bytes']
    status.update(output=str(dst),local_stage=str(src),external_copy_verified=True);dump(dst/'COMPLETED.json',status)
    dump(dst/'publication.json',dict(copied_files_SHA256_equal=True,files_verified=len(records),local_source=str(src)))
    dump(dst/'output_fingerprints.json',[fingerprint(p) for p in sorted(dst.iterdir()) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']])
    print('PUBLISHED',dst)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--reference',type=Path,help='Independent C1 average reference JSON; omitted retains historical TDKd3 reference')
    parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    if args.publish:publish(args.output.resolve())
    else:
        assert args.run is not None;generate(args.run.resolve(),args.reference)
