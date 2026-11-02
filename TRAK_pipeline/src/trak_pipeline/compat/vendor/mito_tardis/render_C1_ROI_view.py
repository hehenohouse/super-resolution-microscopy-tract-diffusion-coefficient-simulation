#!/usr/bin/env python3
"""Extract the verified C1/mask panel and black out pixels outside the cell ROI.

Uses cached RGB viewing pixels only. Does not process scientific image, mask,
localization, or TARDIS inputs. Native pixel-center ROI membership is replicated
at the existing 2x display scale; no new red contour is drawn at the ROI cut.
"""
import argparse
import csv
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image,ImageDraw,ImageFont
import tifffile

from render_point_stack import HERE,PROJECT,dump,fingerprint,pixel_hash,TIFF_NAME,Tee

EXAMPLES=[200,1200,2000]
TARGET='C1_W5min32_ROI_only_frames0000_2499.ome.tif'


def perform(args,out):
    run=args.run
    assert json.loads((run/'COMPLETED.json').read_text())['delivery_checks_passed']
    source_params=json.loads((run/'parameters.json').read_text())
    mask_dir=Path(source_params['mask_directory'])
    source=run/TIFF_NAME
    source_manifest={r['path']:r['sha256'] for r in json.loads((run/'output_fingerprints.json').read_text())}
    expected={r['path']:r['sha256'] for r in json.loads((run/'input_fingerprints_before.json').read_text())}
    expected.update(source_manifest)
    inputs=[source,run/'parameters.json',run/'panel_layout.json',run/'frame_manifest.csv',
            mask_dir/'roi_mask.npy',mask_dir/'parameters.json',mask_dir/'frame_windows_and_status.csv']
    before=[]
    for path in inputs:
        print('SOURCE',path.name,flush=True)
        row=fingerprint(path);assert row['sha256']==expected[row['path']]
        before.append(row)
    dump(out/'input_fingerprints_before.json',before)
    roi=np.load(mask_dir/'roi_mask.npy')
    assert roi.shape==(684,428) and roi.dtype==bool and roi.sum()==32179
    layout=json.loads((run/'panel_layout.json').read_text())
    assert len(layout['panel_origins'])==6
    crop=source_params['crop_xyxy_px'];scale=source_params['display_scale']
    assert crop==[82,232,347,430] and scale==2
    x0,y0,x1,y1=crop;w,h=layout['panel_width'],layout['panel_height']
    assert w==(x1-x0)*scale and h==(y1-y0)*scale
    ys,xs=np.where(roi)
    assert xs.min()>=x0 and xs.max()<x1 and ys.min()>=y0 and ys.max()<y1
    roi_display=np.repeat(np.repeat(roi[y0:y1,x0:x1],scale,0),scale,1)
    px,py=layout['panel_origins'][4]
    mask_params=json.loads((mask_dir/'parameters.json').read_text())
    assert mask_params['temporal_window']==5 and mask_params['min_keep_area_pixel2']==32
    windows=pd.read_csv(mask_dir/'frame_windows_and_status.csv')
    frame_source=pd.read_csv(run/'frame_manifest.csv')
    np.testing.assert_array_equal(frame_source.frame,np.arange(2500))
    np.testing.assert_array_equal(windows.frame,frame_source.frame)
    np.testing.assert_array_equal(frame_source.mask_temporal_applied,windows.temporal_applied)
    np.testing.assert_array_equal(frame_source.loc[~frame_source.mask_temporal_applied,'frame'],[0,1,2498,2499])
    header=64;cfg=dict(source_viewing_stack=str(source),source_panel_index_zero_based=4,
        source_C1=source_params['C1_file'],mask_directory=str(mask_dir),
        original_frames=[0,2499],frames=2500,frame_interval_s=.02,native_pixel_size_nm=117,
        crop_xyxy_native_px=crop,display_scale=scale,display_pixel_size_nm=58.5,
        display_C1_ADU=[407,651],ROI_rule='Original native pixel-center ROI mask, replicated 2x; all outside-ROI image pixels exactly black',
        mask_contour='Retain cached red full-mask contour computed before ROI clipping; do not outline the ROI-clipped binary mask',
        ROI_outline='Cached thin gray cell ROI, clipped to the same inside-pixel rule',
        header_height_display_px=header,output_shape=[2500,header+h,w,3],axes='TYXS',
        global_physical_pixel_size_written=False,
        view_only=True,source_intensities='Existing RGB8 viewing pixels; original16-bit TIFF unchanged',
        resegmentation=False,localizations=False,TARDIS=False)
    dump(out/'parameters.json',cfg)
    dump(out/'software_versions.json',dict(python=sys.version,packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','Pillow','tifffile']}))
    shutil.copyfile(mask_dir/'roi_mask.npy',out/'source_ROI_mask.npy')
    shutil.copyfile(run/'input_fingerprints_before.json',out/'cached_source_provenance.json')
    for name in ['render_C1_ROI_view.py','render_point_stack.py','point_stack_core.py']:
        shutil.copyfile(HERE/name,out/'source_snapshot'/name)
    shutil.copyfile(PROJECT/'picasso_pilot/requirements.lock.txt',out/'requirements.lock.txt')
    (out/'command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--run',str(run)])+'\n')
    font=ImageFont.truetype(source_params['font_path'],15)
    font_small=ImageFont.truetype(source_params['font_path'],13)
    rows=[];readback=[];started=time.perf_counter();target=out/TARGET
    fields=['page_index','frame','time_s','mask_temporal_applied','ROI_native_pixels',
            'source_viewing_page_sha256','source_C1_raw_page_sha256','ROI_image_sha256','rendered_sha256']
    with tifffile.TiffFile(source) as tf:
        assert tf.series[0].axes=='TYXS' and tf.series[0].shape==(2500,530,3292,3)
        assert len(tf.pages)==2500
        def render(f):
            original=tf.pages[f].asarray()
            assert pixel_hash(original)==frame_source.rendered_sha256.iloc[f]
            panel=original[py:py+h,px:px+w]
            clipped=np.where(roi_display[...,None],panel,0).astype(np.uint8)
            assert not clipped[~roi_display].any()
            np.testing.assert_array_equal(clipped[roi_display],panel[roi_display])
            image=Image.new('RGB',(w,header+h),'black');image.paste(Image.fromarray(clipped),(0,header))
            draw=ImageDraw.Draw(image)
            draw.text((8,5),f'TDKd3 T1 | C1 raw + W5/min32 mask | Cell ROI only',font=font,fill='white')
            draw.text((8,25),f'Original frame {f:04d} | {f*.02:.2f} s | 407-651 ADU | display 2x',font=font_small,fill='#DDDDDD')
            status='UNSMOOTHED MIN32 FALLBACK' if not frame_source.mask_temporal_applied.iloc[f] else 'W5 temporal majority + min32'
            draw.text((8,43),status,font=font_small,fill='#F7BD58' if not frame_source.mask_temporal_applied.iloc[f] else '#DDDDDD')
            array=np.asarray(image)
            info=dict(page_index=f,frame=f,time_s=f*.02,mask_temporal_applied=bool(frame_source.mask_temporal_applied.iloc[f]),
                ROI_native_pixels=int(roi.sum()),source_viewing_page_sha256=frame_source.rendered_sha256.iloc[f],
                source_C1_raw_page_sha256=frame_source.raw_C1_page_sha256.iloc[f],
                ROI_image_sha256=pixel_hash(clipped),rendered_sha256=pixel_hash(array))
            return array,info
        for f in EXAMPLES:
            array,info=render(f)
            Image.fromarray(array).save(out/'previews'/f'frame_{f:04d}_C1_ROI.png',compress_level=6)
            print('PREVIEW',f,flush=True)
        with (out/'frame_manifest.csv').open('w',newline='') as csvfile,tifffile.TiffWriter(target,bigtiff=True,ome=True) as writer:
            csvout=csv.DictWriter(csvfile,fieldnames=fields);csvout.writeheader()
            def pages():
                for f in range(2500):
                    array,info=render(f);rows.append(info);csvout.writerow(info)
                    yield array
                    if (f+1)%250==0 and f<2499:
                        csvfile.flush();writer.filehandle.flush()
                        cp=dict(frames=f+1,elapsed_s=time.perf_counter()-started,TIFF_bytes=target.stat().st_size,not_finalized=True)
                        dump(out/'checkpoints'/f'rendered_{f+1:04d}.json',cp)
                        print('CHECKPOINT',json.dumps(cp),flush=True)
            writer.write(pages(),shape=(2500,header+h,w,3),dtype='uint8',photometric='rgb',planarconfig='contig',
                compression='deflate',compressionargs={'level':6},rowsperstrip=128,maxworkers=2,
                metadata=dict(axes='TYXS',Name='TDKd3 T1 C1 W5min32 ROI-only viewing time stack',
                    TimeIncrement=.02,TimeIncrementUnit='s',Description='RGB viewing composite with header; original cell ROI only, outside black. '
                    'Native crop x82:347 y232:430, display2x,117nm/nativepixel. Not a binary measurement mask or raw16-bit image.'))
    assert len(rows)==2500
    dump(out/'checkpoints/rendered_2500.json',dict(frames=2500,elapsed_s=time.perf_counter()-started,TIFF_bytes=target.stat().st_size,not_finalized=False))
    print('CHECKPOINT',2500,flush=True)
    with tifffile.TiffFile(target) as tf:
        assert tf.is_bigtiff and tf.is_ome and len(tf.pages)==2500
        assert tf.series[0].axes=='TYXS' and tf.series[0].shape==(2500,header+h,w,3)
        pixels=ET.fromstring(tf.ome_metadata).find('.//{*}Pixels')
        assert pixels.get('SizeT')=='2500' and pixels.get('SizeZ')=='1' and float(pixels.get('TimeIncrement'))==.02
        for f,page in enumerate(tf.pages):
            image=page.asarray()
            assert pixel_hash(image)==rows[f]['rendered_sha256']
            assert not image[header:][~roi_display].any()
            assert page.keyframe.photometric.name=='RGB' and page.dtype==np.dtype('uint8')
            if f in EXAMPLES:
                with Image.open(out/'previews'/f'frame_{f:04d}_C1_ROI.png') as png:np.testing.assert_array_equal(image,np.asarray(png))
            if f in [0,2499]:Image.fromarray(image).save(out/'previews'/f'frame_{f:04d}_endpoint.png',compress_level=6)
            readback.append(dict(frame=f,decoded_pixels_equal=True,outside_ROI_exactly_black=True))
            if (f+1)%250==0:print('READBACK',f+1,flush=True)
        (out/'OME_metadata.xml').write_text(tf.ome_metadata)
    pd.DataFrame(readback).to_csv(out/'readback_validation.csv',index=False)
    after=[fingerprint(p) for p in inputs];assert before==after
    dump(out/'input_fingerprints_after.json',after)
    dump(out/'validation.json',dict(all2500_source_page_hashes_match=True,all2500_decoded_pages_match=True,
        all_ROI_interior_viewing_pixels_identical_to_cached_C1_panel=True,all_outside_ROI_image_pixels_black=True,
        crop_contains_complete_cell_ROI=True,native_ROI_pixels=int(roi.sum()),original_frame_mapping_preserved=True,
        source_fingerprints_unchanged=True,red_contour_not_recomputed_after_ROI_clipping=True,
        fallback_frames=[0,1,2498,2499],visual_review_pending=True))
    (out/'README.txt').write_text(
        'C1 raw + W5/min32 full-mask red contours, cell ROI only.2500 original time frames0-2499,20ms/frame.\n'
        'Extract cached verified six-panel viewing TIFF panel5; retain all its pixels inside original native ROI, black outside.\n'
        'Native crop x[82,347),y[232,430),display2x nearest pixels; image530x396 plus64px header, total530x460 RGB8.\n'
        'Fixed C1 display407-651ADU. Source raw16-bit and binary masks unchanged. Red contours computed before ROI clipping; no red ROI cut-edge is added.\n'
        'Thin gray ROI outline is retained where inside ROI. The source ROI covers32179 native pixels.\n'
        'Original frame/time header and unsmoothed min32 fallback0/1/2498/2499 labels retained. T is time,Z=1.\n'
        'Deflate lossless OME-BigTIFF. Viewing artifact only, not raw intensity or binary measurement mask.\n'
        'No localization dots, refitting, segmentation, classification, tracking, or TARDIS.\n'
        'Native117nm/pixel; display58.5nm/pixel within image panel. Header prevents global physical calibration of entire composite.\n'
        'All2500 source page hashes, ROI clipping, output readback, original frame mapping and protected input fingerprints validated.\n')
    dump(out/'COMPLETED.json',dict(complete=True,frames=2500,TIFF=str(target),TIFF_bytes=target.stat().st_size,visual_review_pending=True))
    dump(out/'runtime.json',dict(elapsed_s=time.perf_counter()-started))
    (HERE/'latest_C1_ROI_view_output.txt').write_text(str(out)+'\n')
    print('COMPLETED',target,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();args.run=args.run.resolve()
    params=json.loads((args.run/'parameters.json').read_text());base=Path(params['C1_file']).parent
    out=base/'analysis_results/mitochondria_ROI_view'/('T1_C1_W5min32_ROIonly_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    for sub in ['previews','checkpoints','source_snapshot']:(out/sub).mkdir()
    with (out/'run.log').open('w') as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout=Tee(stdout,log);sys.stderr=Tee(stderr,log)
        try:perform(args,out)
        finally:sys.stdout,sys.stderr=stdout,stderr
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file()
        and p.name not in ['output_fingerprints.json','COMPLETED.json']])


if __name__=='__main__':main()
