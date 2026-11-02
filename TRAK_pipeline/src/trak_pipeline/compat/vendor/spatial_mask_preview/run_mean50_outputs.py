#!/usr/bin/env python3
"""Mean50-mask classification and viewing outputs, isolated from TARDIS.

Run with the existing Picasso environment. Exact geometry is delegated to the
existing cleanup environment. Both environments and scientific inputs remain
unchanged. --prepare-only makes the representative previews for visual review;
continue with the same --config and --output directory.
"""
import argparse
import csv
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont,PngImagePlugin
import tifffile

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
sys.path.insert(0,str(PROJECT/'mito_tardis'))
from render_point_stack import dump,fingerprint,pixel_hash,roi_reader,mat_vertices,Tee
from point_stack_core import (gray_crop,mask_contour_alpha,roi_alpha,composite,combined_layer,
                               display_centers,marker_alpha,ON,OFF,ROI_COLOR,MASK_COLOR)
from render_time_projection import count_grid
from mean50_contract import frame_to_block,block_metadata,validate_mapping

EXAMPLES=[200,1200,2000]
TIFF_NAME='C1_mean50_min32_holes_lt64_ROI_only_frames0000_2499.ome.tif'
UNKNOWN=np.array([168,176,186],float)


def c_label(c):
    return c.get('sample_name','TDKd3 T1')


def pointer(name):
    return Path((PROJECT/name).read_text().strip()).resolve()


def resolve(config_path):
    c=json.loads(config_path.read_text())
    if c.get('input_kind')=='annular_cache':
        cache=Path(c['localization_cache']);summary=json.loads((cache/'summary.json').read_text())
        c.update(config_path=str(config_path),audit_source=str(cache/'all_ROI_target_fits.csv'),
            selected_reference=str(cache/'all_ROI_target_fits.csv'),previous_projection=None,old_mask_directory=None,
            font_path='/System/Library/Fonts/Supplemental/Arial.ttf',
            cleanup_python=str((PROJECT/c['cleanup_python']).absolute()),
            expected_ROI_candidates=summary['ROI_candidates'],expected_selected=summary['selected'],
            expected_QC_excluded=summary['QC_excluded'])
        c['new_mask_sha256']=json.loads((Path(c['mask_directory'])/'output_fingerprints.json').read_text())['clean_full_mask.npy']
        return c
    stack=pointer(c['point_stack_pointer'])
    params=json.loads((stack/'parameters.json').read_text())
    comp=pointer(c['comparison_pointer']);projection=pointer(c['projection_pointer'])
    c.update(config_path=str(config_path),point_stack=str(stack),comparison_directory=str(comp),
        previous_projection=str(projection),audit_source=str(comp/'all_ROI_candidates_mask_comparison.csv'),
        selected_reference=str(stack/'point_source.csv'),C1_file=params['C1_file'],ROI_file=params['ROI_file'],
        MAT_file=params['MAT_file'],MAT_record=params['MAT_record'],old_mask_directory=params['mask_directory'],
        # Keep the venv executable path: resolving its symlink bypasses the
        # existing environment and loses its installed geometry dependencies.
        font_path=params['font_path'],cleanup_python=str((PROJECT/c['cleanup_python']).absolute()))
    manifest=json.loads((Path(c['mask_directory'])/'output_fingerprints.json').read_text())
    c['new_mask_sha256']=manifest['clean_full_mask.npy']
    c.update(sample_name='TDKd3 T1',expected_roi_pixels=32179,expected_roi_vertices=454,expected_added_ROI_pixels=3236)
    return c


def expected_inputs(c):
    if c.get('input_kind')=='annular_cache':
        cache=Path(c['localization_cache']);m=Path(c['mask_directory'])
        known={r['path']:r['sha256'] for r in json.loads((cache/'output_fingerprints.json').read_text())}
        known.update({r['path']:r['sha256'] for r in json.loads((cache/'input_fingerprints_before.json').read_text())})
        known.update({str(m/n):h for n,h in json.loads((m/'output_fingerprints.json').read_text()).items()})
        paths=[Path(c['config_path']),Path(c['audit_source']),cache/'config.json',cache/'frame_counts.csv',cache/'summary.json',
            cache/'source_page_manifest.json',Path(c['C1_file']),Path(c['ROI_file']),Path(c['MAT_file']),Path(c['font_path'])]
        paths += [m/n for n in ['mean_images.npy','clean_full_mask.npy','analysis_ROI_mask.npy','original_clean_full_mask.npy',
            'filled_holes_full_mask.npy','roi_mask.npy','roi_vertices_xy.npy','original_frame_numbers_by_block.npy',
            'block_completed.npy','block_frame_mapping.csv','parameters.json','COMPLETE.json','per_block_statistics.csv']]
        return paths,known
    stack=Path(c['point_stack']);comp=Path(c['comparison_directory']);projection=Path(c['previous_projection']);m=Path(c['mask_directory'])
    known={}
    for folder in [stack,comp,projection]:
        known.update({r['path']:r['sha256'] for r in json.loads((folder/'output_fingerprints.json').read_text())})
    known.update({r['path']:r['sha256'] for r in json.loads((stack/'input_fingerprints_before.json').read_text())})
    known.update({str(m/n):h for n,h in json.loads((m/'output_fingerprints.json').read_text()).items()})
    paths=[Path(c['config_path']),Path(c['audit_source']),Path(c['selected_reference']),
        Path(c['C1_file']),Path(c['ROI_file']),Path(c['MAT_file']),Path(c['font_path']),
        stack/'parameters.json',stack/'frame_manifest.csv',projection/'localization_count_maps.npz',
        Path(c['old_mask_directory'])/'roi_mask.npy']
    paths += [m/n for n in ['mean_images.npy','clean_full_mask.npy','analysis_ROI_mask.npy',
        'original_clean_full_mask.npy','filled_holes_full_mask.npy','roi_mask.npy','roi_vertices_xy.npy',
        'original_frame_numbers_by_block.npy','block_completed.npy','block_frame_mapping.csv',
        'parameters.json','COMPLETE.json','per_block_statistics.csv']]
    return paths,known


def verify(c,out):
    height,width=c.get('image_shape_yx',[684,428]);image_shape=(height,width)
    cache=Path(c['localization_cache'] if c.get('input_kind')=='annular_cache' else c['point_stack'])
    assert json.loads((cache/'COMPLETED.json').read_text())['delivery_checks_passed']
    paths,expected=expected_inputs(c);before=[]
    for p in paths:
        print('SOURCE',p.name,flush=True);item=fingerprint(p)
        if str(p) in expected:assert item['sha256']==expected[str(p)],str(p)
        else:assert p.name in ['COMPLETE.json',Path(c['config_path']).name] or str(p)==c['font_path'],str(p)
        before.append(item)
    if (out/'input_fingerprints_before.json').exists():
        assert json.loads((out/'input_fingerprints_before.json').read_text())==before
    dump(out/'input_fingerprints_before.json',before)
    m=Path(c['mask_directory']);params=json.loads((m/'parameters.json').read_text());done=json.loads((m/'COMPLETE.json').read_text())
    assert done['all_passed'] and done['blocks']==50 and done['source_frames']==2500
    assert params['shape']==[50,height,width] and params['axes']=='TYX' and params['pixel_size_um']==.117
    assert params['averaged_frame_interval_s']==1. and params['original_frame_interval_s']==.02
    assert params['minimum_foreground_component_area_pixel2']==32 and params['hole_area_threshold_exclusive_pixel2']==64
    assert params['foreground_connectivity']==8 and params['background_connectivity']==4 and params['no_additional_mask_voting']
    assert np.load(m/'block_completed.npy').all()
    with (m/'block_frame_mapping.csv').open() as f:mapping=list(csv.DictReader(f))
    validate_mapping(mapping,np.load(m/'original_frame_numbers_by_block.npy'))
    full=np.load(m/'clean_full_mask.npy',mmap_mode='r');analysis=np.load(m/'analysis_ROI_mask.npy',mmap_mode='r')
    original=np.load(m/'original_clean_full_mask.npy',mmap_mode='r');filled=np.load(m/'filled_holes_full_mask.npy',mmap_mode='r')
    for a in [full,analysis,original,filled]:assert a.shape==(50,height,width) and a.dtype==bool
    means=np.load(m/'mean_images.npy',mmap_mode='r');assert means.shape==(50,height,width) and means.dtype==np.float32
    roi=np.load(m/'roi_mask.npy');assert roi.shape==image_shape and roi.dtype==bool and roi.sum()==c['expected_roi_pixels']
    if c.get('old_mask_directory'):np.testing.assert_array_equal(roi,np.load(Path(c['old_mask_directory'])/'roi_mask.npy'))
    np.testing.assert_array_equal(analysis,full&roi)
    assert not (original&~full).any();np.testing.assert_array_equal(filled,full&~original)
    assert int((filled&roi).sum())==c['expected_added_ROI_pixels']
    lib,path=roi_reader();vertices=lib.load_imagej_roi(c['ROI_file'])
    np.testing.assert_array_equal(vertices,np.load(m/'roi_vertices_xy.npy'))
    np.testing.assert_array_equal(vertices,mat_vertices(c['MAT_file'],c['MAT_record']));assert len(vertices)==c['expected_roi_vertices']
    with tifffile.TiffFile(c['C1_file']) as tf:
        assert len(tf.pages)==2500 and tf.series[0].axes=='TYX' and tf.series[0].shape==(2500,height,width)
        assert tf.pages[0].dtype==np.dtype('uint16')
    assert c['k']==1 and c['q_tolerance']==1e-12 and c['block_size_frames']==50
    assert c['frame_count']==2500 and c['pixel_size_nm']==117 and c['frame_interval_s']==.02
    x0,y0,x1,y1=c['crop_xyxy_px'];ys,xs=np.where(roi)
    assert 0<=x0<=xs.min() and xs.max()<x1<=width and 0<=y0<=ys.min() and ys.max()<y1<=height
    assert c['C1_display_ADU']==[407,651]
    dump(out/'source_validation.json',dict(all_checks_passed=True,ROI_MAT_and_mask_vertices_identical=True,
        final_holefilled_full_mask_verified=True,foreground_min32_holes_strictly_lt64=True,
        exact50_blocks_original_frames0_2499=True,all50_blocks_completed=True,ROI_pixels=int(roi.sum()),
        added_ROI_pixels=int((filled&roi).sum()),registration='exploratory_identity_registration_unverified',
        mask_selected_as_final_flag_unchanged=done.get('selected_as_final')))
    shutil.copyfile(path,out/'source_snapshot/roi_reader.py')
    return full,roi,vertices,before


def child(command,log_path):
    with log_path.open('w') as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
        for line in proc.stdout:print(line,end='',flush=True);log.write(line);log.flush()
        if proc.wait():raise RuntimeError('Child command failed: '+shlex.join(command))


def preflight(c,out):
    for pattern in ['test_geometry.py','test_mean50_contract.py','test_mean50_cache_interface.py']:
        command=[c['cleanup_python'],'-B','-m','unittest','discover','-s',str(HERE),'-p',pattern,'-v']
        result=subprocess.run(command,capture_output=True,text=True)
        (out/(pattern+'.log')).write_text(result.stdout+result.stderr)
        if result.returncode:raise RuntimeError(result.stdout+result.stderr)
    print('PREFLIGHT geometry and mean50 mapping tests passed',flush=True)


class C1View:
    def __init__(self,c,full,roi,vertices,source_pages):
        self.c=c;self.full=full;self.crop=c['crop_xyxy_px'];self.scale=2;self.header=64
        self.w=(self.crop[2]-self.crop[0])*2;self.h=(self.crop[3]-self.crop[1])*2;self.source_pages=source_pages
        x0,y0,x1,y1=self.crop
        self.roi=np.repeat(np.repeat(roi[y0:y1,x0:x1],2,0),2,1)
        self.outline=roi_alpha(vertices,self.crop,2)
        self.contours={b:mask_contour_alpha(full[b],self.crop,2) for b in range(50)}
        self.fonts={s:ImageFont.truetype(c['font_path'],s) for s in [13,14]}

    def render(self,f,raw):
        assert raw.shape==tuple(self.c.get('image_shape_yx',[684,428])) and raw.dtype==np.uint16
        if self.source_pages is not None:assert pixel_hash(raw)==self.source_pages.raw_C1_page_sha256.iloc[f]
        b=frame_to_block(f);gray=gray_crop(raw,self.crop,2,[407,651]);base=np.repeat(gray[...,None],3,axis=2)
        panel=composite(composite(base,self.contours[b],MASK_COLOR),self.outline,ROI_COLOR)
        panel[~self.roi]=0;assert not panel[~self.roi].any()
        untouched=self.roi&(self.contours[b]==0)&(self.outline==0)
        np.testing.assert_array_equal(panel[untouched],base[untouched])
        image=Image.new('RGB',(self.w,self.header+self.h),'black');image.paste(Image.fromarray(panel),(0,self.header))
        draw=ImageDraw.Draw(image)
        draw.text((8,5),c_label(self.c)+' | C1 raw + mean50/min32/holes<64 | ROI only',font=self.fonts[14],fill='white')
        draw.text((8,25),f'Original frame {f:04d} | {f*.02:.2f} s | 407-651 ADU | display 2x',font=self.fonts[13],fill='#DDDDDD')
        draw.text((8,44),f'Mask block {b:02d} | frames {50*b}-{50*b+49} | {b:.2f}-{b+1:.2f} s average',font=self.fonts[13],fill='#F7BD58')
        array=np.asarray(image)
        info=dict(frame=f,time_s=f*.02,**block_metadata(f),raw_C1_page_sha256=pixel_hash(raw),
            mask_page_sha256=pixel_hash(self.full[b]),rendered_sha256=pixel_hash(array))
        return array,info


def preview_images(c,out,view):
    records=[]
    with tifffile.TiffFile(c['C1_file']) as tf:
        for f in EXAMPLES:
            image,info=view.render(f,tf.pages[f].asarray())
            Image.fromarray(image).save(out/'previews'/f'frame_{f:04d}_C1_ROI.png',compress_level=6)
            records.append(info);print('PREVIEW',f,'mask block',frame_to_block(f),flush=True)
    dump(out/'previews/image_manifest.json',records)


def project_points(c,out,roi,vertices):
    path=out/c.get('selected_classification_csv','classification/selected_localizations_mean50.csv')
    label_field=c.get('classification_label_field','mean50_pointwise_label')
    recipe=c.get('mask_recipe','mean50_min32_holes_lt64')
    recipe_display=c.get('mask_recipe_display','mean50/min32/holes<64')
    columns=['candidate_index','frame','fit_x_px','fit_y_px','x_se_nm','y_se_nm',label_field]
    table=pd.read_csv(path,usecols=columns,float_precision='round_trip');n=c['expected_selected'];assert len(table)==n
    assert table.candidate_index.is_unique and np.isfinite(table[['fit_x_px','fit_y_px','x_se_nm','y_se_nm']]).all().all()
    assert table[['x_se_nm','y_se_nm']].gt(0).all().all()
    names=['On','Not_on','unknown'];labels=['mito-compatible','non-mito-compatible','unknown']
    crop=c['crop_xyxy_px'];scale=4;w,h=(crop[2]-crop[0])*4,(crop[3]-crop[1])*4;alpha=[];maps=[];numbers=[]
    for label in labels:
        rows=table[table[label_field]==label];xy=rows[['fit_x_px','fit_y_px']].to_numpy()
        numbers.append(len(rows));alpha.append(marker_alpha((h,w),display_centers(xy,crop,4),.2*4));maps.append(count_grid(xy,crop,4))
    assert sum(numbers)==n
    all_map=count_grid(table[['fit_x_px','fit_y_px']].to_numpy(),crop,4)
    np.testing.assert_array_equal(all_map,maps[0]+maps[1]+maps[2])
    if c.get('previous_projection'):
        with np.load(Path(c['previous_projection'])/'localization_count_maps.npz') as old:
            np.testing.assert_array_equal(all_map,old['all_valid'])
    else:
        original=pd.read_csv(c['audit_source'],usecols=['fit_x_px','fit_y_px','qc_unflagged_model_fit','fitted_center_inside_ROI'],float_precision='round_trip')
        original=original[original.qc_unflagged_model_fit&original.fitted_center_inside_ROI]
        assert len(original)==n
        np.testing.assert_array_equal(all_map,count_grid(original[['fit_x_px','fit_y_px']].to_numpy(),crop,4))
    np.savez_compressed(out/'projections/localization_count_maps.npz',On=maps[0],Not_on=maps[1],unknown=maps[2],
        all_valid=all_map,native_crop_xyxy_px=np.array(crop),nm_per_bin=np.array(29.25))
    base=composite(np.zeros((h,w,3),np.uint8),roi_alpha(vertices,crop,4),ROI_COLOR)
    third=combined_layer(base,alpha[0],alpha[1])
    if numbers[2]:
        total=alpha[0]+alpha[1]+alpha[2];color=np.zeros((h,w,3),float)
        np.divide(alpha[0][...,None]*ON+alpha[1][...,None]*OFF+alpha[2][...,None]*UNKNOWN,total[...,None],out=color,where=total[...,None]>0)
        third=composite(base,np.maximum.reduce(alpha),color)
    panels=[composite(base,alpha[0],ON),composite(base,alpha[1],OFF),third]
    filenames=['on_all_timepoints','not_on_all_timepoints','all_valid_all_timepoints']
    titles=[f'On | {numbers[0]:,} localizations',f'Not-on | {numbers[1]:,} localizations',
            f'All valid | {n:,} localizations | unknown {numbers[2]:,}']
    images=[];font=lambda s:ImageFont.truetype(c['font_path'],s)
    for i,(panel,name,title) in enumerate(zip(panels,filenames,titles)):
        image=Image.new('RGB',(w,h+128),'black');image.paste(Image.fromarray(panel),(0,80));draw=ImageDraw.Draw(image)
        draw.text((12,8),c_label(c)+' | frames 0-2499 pooled | '+recipe_display,font=font(22),fill='white')
        draw.text((12,43),title,font=font(24),fill=tuple(ON.astype(int)) if i==0 else tuple(OFF.astype(int)) if i==1 else 'white')
        bx,by=display_centers([[crop[0]+14,crop[3]-6]],crop,4)[0]
        draw.line([(bx,80+by),(bx+2000/117*4,80+by)],fill='white',width=3)
        draw.text((bx,80+by-23),'2 um',font=font(17),fill='white')
        draw.text((12,h+88),c.get('projection_footer','Each 20 ms localization uses its 1 s mean-image mask; repeat observations retained.'),font=font(18),fill='#DDDDDD')
        draw.text((12,h+111),'Cell ROI | C1/C2 identity registration unverified | Dots mark centers, not errors.',font=font(17),fill='#BBBBBB')
        meta=PngImagePlugin.PngInfo();meta.add_text('mask_recipe',recipe);meta.add_text('localizations',str([numbers[0],numbers[1],n][i]))
        image.save(out/'projections'/f'{name}.png',pnginfo=meta,compress_level=6)
        tifffile.imwrite(out/'projections'/f'{name}.tif',np.asarray(image),photometric='rgb',compression='deflate',metadata={'axes':'YXS'})
        np.testing.assert_array_equal(tifffile.imread(out/'projections'/f'{name}.tif'),np.asarray(image));images.append(image)
    overview=Image.new('RGB',(3*w+64,h+128),'black')
    for i,image in enumerate(images):overview.paste(image,(i*(w+32),0))
    overview.save(out/'projections/all_timepoints_triptych.png',compress_level=6)
    dump(out/'projections/validation.json',dict(all_selected_points_retained=True,selected=n,source_all_valid_count_grid_reproduced_exactly=True,
        group_count_maps_sum_to_all_points=True,PNG_TIFF_pixels_identical=True,On=numbers[0],Not_on=numbers[1],unknown=numbers[2]))
    print('PROJECTION',json.dumps(dict(zip(names,numbers))),flush=True)


def export_stack(c,out,view):
    target=out/'ROI_C1_stack'/TIFF_NAME;assert not target.exists()
    rows=[];start=time.perf_counter()
    with (tifffile.TiffFile(c['C1_file']) as source,
          (out/'ROI_C1_stack/frame_manifest.csv').open('w',newline='') as log,
          tifffile.TiffWriter(target,bigtiff=True,ome=True) as writer):
        fields=['page_index','frame','time_s',*block_metadata(0),'raw_C1_page_sha256','mask_page_sha256','rendered_sha256']
        csvout=csv.DictWriter(log,fieldnames=fields);csvout.writeheader()
        def pages():
            for f in range(2500):
                image,info=view.render(f,source.pages[f].asarray());info=dict(page_index=f,**info)
                rows.append(info);csvout.writerow(info);yield image
                if (f+1)%250==0 and f<2499:
                    log.flush();writer.filehandle.flush()
                    cp=dict(rendered_frames=f+1,elapsed_s=time.perf_counter()-start,TIFF_bytes=target.stat().st_size,not_finalized=True)
                    dump(out/'checkpoints'/f'rendered_{f+1:04d}.json',cp);print('RENDER_CHECKPOINT',json.dumps(cp),flush=True)
        writer.write(pages(),shape=(2500,view.header+view.h,view.w,3),dtype='uint8',photometric='rgb',planarconfig='contig',
            compression='deflate',compressionargs={'level':6},rowsperstrip=128,maxworkers=2,
            metadata=dict(axes='TYXS',Name=c_label(c)+' C1 raw ROI mean50 min32 holes lt64',TimeIncrement=.02,TimeIncrementUnit='s',
                Description='Viewing composite with64px header. Original20ms C1 frames, mask f//50 from1s average. '
                    f'Native117nm/pixel, crop {c["crop_xyxy_px"]} display2x. RGB8, not scientific intensity or binary mask.'))
    assert len(rows)==2500
    dump(out/'checkpoints/rendered_2500.json',dict(rendered_frames=2500,elapsed_s=time.perf_counter()-start,TIFF_bytes=target.stat().st_size,not_finalized=False))
    print('RENDER_CHECKPOINT 2500',flush=True)
    decoded=[]
    with tifffile.TiffFile(target) as tf:
        assert tf.is_ome and tf.is_bigtiff and len(tf.pages)==2500 and tf.series[0].axes=='TYXS'
        assert tf.series[0].shape==(2500,view.header+view.h,view.w,3)
        pixels=ET.fromstring(tf.ome_metadata).find('.//{*}Pixels')
        assert pixels.get('SizeT')=='2500' and pixels.get('SizeZ')=='1' and pixels.get('Type')=='uint8'
        assert float(pixels.get('TimeIncrement'))==.02 and pixels.get('TimeIncrementUnit')=='s'
        for f,page in enumerate(tf.pages):
            image=page.asarray();assert pixel_hash(image)==rows[f]['rendered_sha256']
            assert not image[64:][~view.roi].any()
            assert page.keyframe.photometric.name=='RGB' and page.keyframe.compression.name in ['DEFLATE','ADOBE_DEFLATE']
            if f in EXAMPLES:
                with Image.open(out/'previews'/f'frame_{f:04d}_C1_ROI.png') as png:np.testing.assert_array_equal(image,np.asarray(png))
            decoded.append(dict(frame=f,block_index=frame_to_block(f),decoded_pixels_equal=True,outside_ROI_black=True))
            if (f+1)%250==0:print('READBACK',f+1,flush=True)
        (out/'ROI_C1_stack/OME_metadata.xml').write_text(tf.ome_metadata)
    pd.DataFrame(decoded).to_csv(out/'ROI_C1_stack/readback_validation.csv',index=False)
    manifest=pd.DataFrame(rows);np.testing.assert_array_equal(manifest.frame,np.arange(2500))
    np.testing.assert_array_equal(manifest.mean50_block_index,np.arange(2500)//50)
    dump(out/'ROI_C1_stack/validation.json',dict(all2500_pages_readback_equal=True,all_original_C1_page_hashes_match=True,
        ROI_exterior_black=True,original_frame_mapping_preserved=True,mask_mapping_frame_div50_verified=True,
        shape=[2500,view.header+view.h,view.w,3],SizeT=2500,SizeZ=1,TimeIncrement_s=.02,RGB8=True,lossless_Deflate=True,
        full_mask_contours_before_ROI_clipping=True,no_old_W5_or_endpoint_fallback_labels=True))
    return target


def archive_sources(c,out):
    for n in ['run_mean50_outputs.py','mean50_classify.py','mean50_contract.py','test_mean50_contract.py','test_mean50_cache_interface.py','geometry.py','test_geometry.py']:
        shutil.copyfile(HERE/n,out/'source_snapshot'/n)
    for n in ['point_stack_core.py','render_point_stack.py','render_time_projection.py']:
        shutil.copyfile(PROJECT/'mito_tardis'/n,out/'source_snapshot'/n)
    shutil.copyfile(Path(c['config_path']),out/'config_mean50_outputs.json')
    shutil.copyfile(PROJECT/'picasso_pilot/requirements.lock.txt',out/'picasso_requirements.lock.txt')
    shutil.copyfile(Path(c['mask_directory'])/'requirements.lock.txt',out/'mask_source_requirements.lock.txt')


def perform(args,c,out):
    full,roi,vertices,before=verify(c,out);preflight(c,out);archive_sources(c,out)
    dump(out/'software_versions.json',dict(python=sys.version,executable=sys.executable,
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','Pillow','tifffile','h5py']}))
    source_pages=None if c.get('input_kind')=='annular_cache' else pd.read_csv(Path(c['point_stack'])/'frame_manifest.csv')
    view=C1View(c,full,roi,vertices,source_pages)
    preview_images(c,out,view)
    worker=[c['cleanup_python'],'-B','-u',str(HERE/'mean50_classify.py'),'--resolved',str(out/'resolved_config.json'),'--output',str(out)]
    if args.prepare_only:
        child(worker+['--preview-only'],out/'previews/classification_worker.log')
        dump(out/'run_status.json',dict(status='previews_ready',frames=EXAMPLES))
        print('PREVIEWS_READY',out,flush=True);return
    child(worker,out/'classification/worker.log')
    project_points(c,out,roi,vertices);target=export_stack(c,out,view)
    print('Rechecking protected inputs',flush=True)
    paths,_=expected_inputs(c);after=[fingerprint(p) for p in paths];assert before==after
    dump(out/'input_fingerprints_after.json',after)
    summary=json.loads((out/'classification/summary.json').read_text())
    assert summary['selected']==c['expected_selected'] and summary['QC_excluded']==c['expected_QC_excluded']
    dump(out/'validation.json',dict(all_implementation_checks_passed=True,source_fingerprints_unchanged=True,
        selected_point_identity_and_fits_unchanged=True,all_original_audit_fields_preserved=True,
        old_W5_outputs_preserved=True,old_all_valid_count_grid_reproduced=True,all2500_TIFF_pages_readback_equal=True,
        no_refitting_resegmentation_or_TARDIS=True,mean_mask_interval_s=1.,raw_C1_interval_s=.02,visual_review_pending=True))
    (out/'README.txt').write_text(
        c_label(c)+': mean50/min32/holes<64 mask-specific spatial classification and viewing outputs.\n'+
        'Use final hole-filled clean_full_mask, not before-hole or ROI-clipped mask, for per-point k1 ellipse contact.\n'
        'Original frame f uses block f//50:50 nonoverlapping1s average-image masks; no W5 voting, interpolation or motion compensation.\n'
        f'Original{c["expected_ROI_candidates"]} ROI candidates preserved verbatim;{c["expected_selected"]} selected current QC/ROI/finite positive SE;{c["expected_QC_excluded"]} QC exclusions not restored.\n'
        'X/Y axes=per-point nominal-model SE_nm/117, diagonal covariance, closed pixel squares, tangency included, q tolerance1e-12.\n'
        'Unknown rule uses original per-frame C1 zero regions/FOV, never pooled into Not-on; centers selected by unchanged common ROI.\n'
        'Any original spatial labels retained when present; mean50 labels appended. No molecule tracking or deduplication.\n'
        f'Projection images: all2500 frames pooled, On green/Not-on purple/unknown gray, all{c["expected_selected"]} points in third panel.\n'
        f'Crop {c["crop_xyxy_px"]},4x display,dot radius0.20nativepx,2um bar; dots are viewing markers, not error ellipses.\n'
        'Count grids conserve all observations and exactly reproduce old all-valid grid. Repeated observations are not unique molecules.\n'
        'ROI C1 stack:2500 original20ms C1 frames,407-651ADU fixed grayscale,red full-mask contours,ROI outside black,2x display.\n'
        'Each64px header records original frame/time and1s mask source block; no W5 or fallback labels.\n'
        f'RGB8 OME-BigTIFF2500x{view.header+view.h}x{view.w}x3,TYXS,T2500/Z1,20ms/page,lossless Deflate; viewing artifact,not raw intensity or binary mask.\n'
        'Native117nm/px,display58.5nm/px within image panel; header prevents global physical calibration of composite.\n'
        'C1/C2 identity registration remains unverified;1s masks are not instantaneous20ms boundaries or evidence of binding/cytoplasm.\n'
        'Original TIFF,mask arrays,localization/QC values,mask default flags and paused47-sample batch remain untouched.\n'
        'No new segmentation,fitting,TARDIS,trajectories,significance,deconvolution or biological-state inference.\n')
    dump(out/'COMPLETED.json',dict(all_implementation_checks_passed=True,visual_review_pending=True,
        output=str(out),counts=summary,TIFF=str(target),TIFF_bytes=target.stat().st_size))
    dump(out/'run_status.json',dict(status='complete_pending_visual_review'))
    output_pointer='latest_mean50_WT_T1_output.txt' if c.get('input_kind')=='annular_cache' else 'latest_mean50_output.txt'
    (HERE/output_pointer).write_text(str(out)+'\n')
    print('COMPLETED',json.dumps(dict(output=str(out),counts=summary,TIFF=str(target))),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--output',type=Path)
    args=ap.parse_args();args.config=args.config.resolve()
    if args.output:
        out=args.output.resolve();c=json.loads((out/'resolved_config.json').read_text())
        assert c['config_path']==str(args.config)
        assert json.loads((out/'run_status.json').read_text())['status']=='previews_ready'
        assert not (out/'ROI_C1_stack'/TIFF_NAME).exists()
    else:
        c=resolve(args.config);base=Path(c['C1_file']).parent
        out=base/c['output_relative_directory']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out.mkdir(parents=True,exist_ok=False)
        for sub in ['source_snapshot','classification','projections','ROI_C1_stack','previews','checkpoints']:(out/sub).mkdir()
        dump(out/'resolved_config.json',c)
    attempt_pointer='latest_mean50_WT_T1_attempt.txt' if c.get('input_kind')=='annular_cache' else 'latest_mean50_attempt.txt'
    (HERE/attempt_pointer).write_text(str(out)+'\n')
    command=shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(args.config),
        *(['--prepare-only'] if args.prepare_only else []),*(['--output',str(out)] if args.output else [])])
    (out/('preview_command.txt' if args.prepare_only else 'run_command.txt')).write_text(command+'\n')
    started=time.perf_counter()
    with (out/'run.log').open('a') as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout=Tee(stdout,log);sys.stderr=Tee(stderr,log)
        try:
            print('OUTPUT',out,flush=True);perform(args,c,out)
        except BaseException:
            dump(out/'FAILED.json',dict(traceback=traceback.format_exc(),failed_utc=datetime.now(timezone.utc).isoformat()));raise
        finally:sys.stdout,sys.stderr=stdout,stderr
    dump(out/('preview_runtime.json' if args.prepare_only else 'runtime.json'),dict(elapsed_s=time.perf_counter()-started))
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file()
        and p.name not in ['output_fingerprints.json','COMPLETED.json']])


if __name__=='__main__':main()
