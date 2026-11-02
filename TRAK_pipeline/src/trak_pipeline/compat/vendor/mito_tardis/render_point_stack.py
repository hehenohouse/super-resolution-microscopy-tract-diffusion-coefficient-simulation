#!/usr/bin/env python3
"""Read-only cached localization/C1/C2/mask six-panel RGB time-stack export."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shlex
import shutil
import sys
import time
import traceback
import unittest
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd
from PIL import Image,PngImagePlugin
import tifffile

from point_stack_core import Renderer,display_centers

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
EXAMPLES=[200,1200,2000]
TIFF_NAME='T1_C1C2_six_panels_frames0000_2499.ome.tif'


class Tee:
    def __init__(self,stream,log):self.stream,self.log=stream,log
    def write(self,s):self.stream.write(s);self.log.write(s);self.log.flush();return len(s)
    def flush(self):self.stream.flush();self.log.flush()


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def fingerprint(path):
    p=Path(path).resolve();h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h.hexdigest())


def pixel_hash(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def roi_reader():
    path=PROJECT/'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py'
    if not path.is_file():path=HERE/'roi_reader.py'
    spec=importlib.util.spec_from_file_location('point_stack_readonly_roi',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module,path


def mat_vertices(path,name):
    with h5py.File(path,'r') as h:
        result=[]
        for ref in h['data'][()].ravel(order='F'):
            if not ref:continue
            r=h[ref]
            if not isinstance(r,h5py.Group) or 'name' not in r:continue
            n=''.join(chr(int(x)) for x in r['name'][()].ravel(order='F') if int(x))
            if n==name:result.append(r['roiData/roiCoordinates'][()].T.astype(float))
        assert len(result)==1
        return result[0]


def load_inputs(run):
    complete=json.loads((run/'COMPLETED.json').read_text());assert complete['delivery_checks_passed']
    source=json.loads((run/'source_reference.json').read_text())
    comparison=Path(source['comparison_directory'])
    verified=json.loads((comparison/'source_verification.json').read_text())
    base=Path(verified['config']['data_directory']);mask=Path(source['mask_directory'])
    assert mask.name=='Nellie_broad_W5_min32_20261014T181412Z'
    cfg=dict(source_run=str(run),C1_file=verified['new']['C1_file'],C2_file=verified['new']['C2_file'],
        mask_directory=str(mask),ROI_file=str(base/verified['config']['roi_filename']),
        MAT_file=verified['config']['mat_file'],MAT_record=verified['config']['mat_record'],
        frames=2500,frame_interval_s=.02,pixel_size_nm=117,crop_xyxy_px=[82,232,347,430],
        display_scale=2,dot_radius_native_px=1.5,circle_radius_native_px=3.,
        display_C1_ADU=[407,651],display_C2_ADU=[406,554],
        font_path='/System/Library/Fonts/Supplemental/Arial.ttf',
        checkpoint_frames=250,registration_status='exploratory_identity_registration_unverified',
        group_rule='Existing k=1 new_pointwise_label; original QC and refined center in common cell ROI',
        probability_filtering=False,refitting=False,reclassification=False,TARDIS=False,
        expected_group_counts={'On':57615,'Not-on':57398},expected_points=115013,
        panels=['On points','Not-on points','On + Not-on points','C2 raw + localization circles','C1 raw + full mask contours',
                'C1 raw + full mask contours + localization circles'],
        combined_panel_background='C1 raw, same fixed grayscale as panel five',
        combined_panel_layer_order=['C1 grayscale','full mask red contour','cell ROI gray outline','On/Not-on circles with commutative overlap'],
        dtype='uint8',axes='TYXS',TIFF='OME-BigTIFF RGB Deflate level6',
        view_only=True,global_spatial_calibration=False,
        display_mapping='display=(native-crop_origin+0.5)*2-0.5; x right, y down',
        marker_antialiasing='analytic floating-center signed distance; one-display-pixel coverage ramp',
        combined_marker_overlap='commutative coverage-weighted color mixture; neither group drawn preferentially on top')
    manifest={r['path']:r['sha256'] for r in json.loads((run/'output_fingerprints.json').read_text())}
    expected={str(run/name):manifest[str(run/name)] for name in
              ['statistical_input.csv','roi_vertices.csv','per_frame_counts.csv','source_reference.json','config.json']}
    paths=[cfg['C1_file'],cfg['C2_file'],cfg['ROI_file'],cfg['MAT_file']]
    paths += [str(mask/n) for n in ['clean_full_mask.npy','roi_vertices_xy.npy','roi_mask.npy',
              'original_frame_numbers.npy','frame_completed.npy','parameters.json','frame_windows_and_status.csv']]
    for p in paths:expected[p]=verified['expected_sha256'][p]
    expected[cfg['font_path']]=fingerprint(cfg['font_path'])['sha256']
    return cfg,expected,base


def verify(run,c,expected,out):
    before=[]
    for path,digest in expected.items():
        print('SOURCE',Path(path).name,flush=True);f=fingerprint(path)
        assert f['sha256']==digest,path;before.append(f)
    if (out/'input_fingerprints_before.json').is_file():
        assert json.loads((out/'input_fingerprints_before.json').read_text())==before
    else:dump(out/'input_fingerprints_before.json',before)
    m=Path(c['mask_directory']);par=json.loads((m/'parameters.json').read_text());done=json.loads((m/'COMPLETE.json').read_text())
    assert done['all_passed'] and done['frames']==2500 and done['summary']['window']==5
    assert done['summary']['min_keep_area_pixel2']==32
    assert par['shape']==[2500,684,428] and par['axes']=='TYX' and par['pixel_size_um']==.117 and par['time_interval_s']==.02
    mask=np.load(m/'clean_full_mask.npy',mmap_mode='r')
    assert mask.shape==(2500,684,428) and mask.dtype==bool
    assert np.load(m/'frame_completed.npy',mmap_mode='r').all()
    np.testing.assert_array_equal(np.load(m/'original_frame_numbers.npy'),np.arange(2500))
    lib,path=roi_reader();vertices=lib.load_imagej_roi(c['ROI_file'])
    np.testing.assert_array_equal(vertices,np.load(m/'roi_vertices_xy.npy'))
    np.testing.assert_array_equal(vertices,pd.read_csv(run/'roi_vertices.csv').to_numpy())
    np.testing.assert_array_equal(vertices,mat_vertices(c['MAT_file'],c['MAT_record']))
    assert len(vertices)==454
    table=pd.read_csv(run/'statistical_input.csv',float_precision='round_trip')
    assert len(table)==115013 and table.candidate_index.is_unique
    assert table.qc_unflagged_model_fit.all() and table.fitted_center_inside_ROI.all() and table.spatial_selected.all()
    assert table.spatial_registration_status.eq(c['registration_status']).all()
    assert table.frame.between(0,2499).all() and np.isfinite(table[['fit_x_px','fit_y_px']]).all().all()
    for group,label in [('On','mito-compatible'),('Not-on','non-mito-compatible')]:
        rows=table[table.tardis_group==group]
        assert len(rows)==c['expected_group_counts'][group] and rows.new_pointwise_label.eq(label).all()
    assert lib.points_in_polygon(table[['fit_x_px','fit_y_px']].to_numpy(),vertices).all()
    counts=pd.read_csv(run/'per_frame_counts.csv');windows=pd.read_csv(m/'frame_windows_and_status.csv')
    np.testing.assert_array_equal(counts.frame,np.arange(2500));np.testing.assert_array_equal(windows.frame,counts.frame)
    np.testing.assert_array_equal(counts.mask_temporal_applied,windows.temporal_applied)
    np.testing.assert_array_equal(counts.loc[~counts.mask_temporal_applied,'frame'],[0,1,2498,2499])
    groups={}
    for g in ['On','Not-on']:
        part=table[table.tardis_group==g]
        np.testing.assert_array_equal(part.groupby('frame').size().reindex(range(2500),fill_value=0),counts[g])
        groups[g]={int(f):p[['fit_x_px','fit_y_px']].to_numpy(float) for f,p in part.groupby('frame')}
    centers=display_centers(table[['fit_x_px','fit_y_px']],c['crop_xyxy_px'],c['display_scale'])
    inverse=(centers+.5)/2+np.array(c['crop_xyxy_px'][:2])-.5
    np.testing.assert_allclose(inverse,table[['fit_x_px','fit_y_px']],rtol=0,atol=1e-12)
    assert (centers>=0).all() and (centers[:,0]<530).all() and (centers[:,1]<396).all()
    for raw in [c['C1_file'],c['C2_file']]:
        with tifffile.TiffFile(raw) as tf:
            assert tf.series[0].axes=='TYX' and tf.series[0].shape==(2500,684,428)
            assert len(tf.pages)==2500 and all(p.shape==(684,428) and p.dtype==np.dtype('uint16') for p in tf.pages)
    dump(out/'source_validation.json',dict(source_checks_passed=True,points=115013,On=57615,Not_on=57398,
        original_QC_ROI_and_labels_unchanged=True,ROI_MAT_mask_vertices_identical=True,
        original_frames=np.arange(2500).tolist(),floating_center_roundtrip_atol_native_px=1e-12,
        fallback_frames=[0,1,2498,2499],no_coordinate_transform=True,registration_verified=False))
    shutil.copyfile(run/'statistical_input.csv',out/'point_source.csv')
    assert fingerprint(out/'point_source.csv')['sha256']==expected[str(run/'statistical_input.csv')]
    coords=table[['candidate_index','frame','tardis_group','fit_x_px','fit_y_px']].copy()
    coords['panel_local_x_display_px']=centers[:,0];coords['panel_local_y_display_px']=centers[:,1]
    coords.to_csv(out/'display_coordinates.csv',index=False)
    for name in ['render_point_stack.py','point_stack_core.py','test_point_stack.py']:
        shutil.copyfile(HERE/name,out/'source_snapshot'/name)
    shutil.copyfile(path,out/'source_snapshot/roi_reader.py')
    shutil.copyfile(PROJECT/'picasso_pilot/requirements.lock.txt',out/'requirements.lock.txt')
    return mask,vertices,groups,counts,before


def frame_image(renderer,f,groups,counts,C1,C2,mask):
    on=groups['On'].get(f,np.empty((0,2)));off=groups['Not-on'].get(f,np.empty((0,2)))
    raw1=C1.pages[f].asarray();raw2=C2.pages[f].asarray();mm=mask[f]
    image,info=renderer.render(f,on,off,raw1,raw2,mm,fallback=not bool(counts.mask_temporal_applied.iloc[f]))
    assert info['On']==counts.On.iloc[f] and info['Not_on']==counts['Not-on'].iloc[f]
    info.update(raw_C1_page_sha256=pixel_hash(raw1),raw_C2_page_sha256=pixel_hash(raw2),
                mask_page_sha256=pixel_hash(mm),rendered_sha256=pixel_hash(image))
    return image,info


def previews(renderer,groups,counts,C1,C2,mask,out):
    result=[]
    for frame in EXAMPLES:
        image,info=frame_image(renderer,frame,groups,counts,C1,C2,mask)
        meta=PngImagePlugin.PngInfo()
        for key,v in info.items():meta.add_text(key,str(v))
        path=out/'previews'/f'frame_{frame:04d}_six_panels.png'
        Image.fromarray(image).save(path,pnginfo=meta,compress_level=6)
        with Image.open(path) as check:np.testing.assert_array_equal(np.asarray(check),image)
        result.append(info)
        print('PREVIEW',json.dumps(info),flush=True)
    dump(out/'preview_manifest.json',result)


def export(renderer,c,groups,counts,C1,C2,mask,out):
    target=out/TIFF_NAME;assert not target.exists(),'Never overwrite an existing TIFF; use a fresh run after failure'
    rows=[];fields=['page_index','frame','time_s','On','Not_on','combined','mask_temporal_applied',
                   'raw_C1_page_sha256','raw_C2_page_sha256','mask_page_sha256','rendered_sha256']
    started=time.perf_counter()
    with (out/'frame_manifest.csv').open('w',newline='') as log,tifffile.TiffWriter(target,bigtiff=True,ome=True) as writer:
        csvlog=csv.DictWriter(log,fieldnames=fields);csvlog.writeheader()
        def pages():
            for f in range(2500):
                image,info=frame_image(renderer,f,groups,counts,C1,C2,mask)
                info=dict(page_index=f,**info);rows.append(info);csvlog.writerow(info)
                yield image
                if (f+1)%250==0 and f+1<2500:
                    log.flush();writer.filehandle.flush()
                    cp=dict(rendered_frames=f+1,On=sum(r['On'] for r in rows),Not_on=sum(r['Not_on'] for r in rows),
                        elapsed_s=time.perf_counter()-started,TIFF_bytes_at_checkpoint=target.stat().st_size,
                        TIFF_not_finalized_until_all_frames_complete=True)
                    dump(out/'checkpoints'/f'rendered_{f+1:04d}.json',cp)
                    dump(out/'run_status.json',dict(status='exporting',**cp))
                    print('CHECKPOINT',json.dumps(cp),flush=True)
        writer.write(pages(),shape=(2500,renderer.height,renderer.width,3),dtype='uint8',
            photometric='rgb',planarconfig='contig',compression='deflate',compressionargs={'level':6},
            maxworkers=2,rowsperstrip=128,extratags=[(274,'H',1,1,False)],
            metadata=dict(axes='TYXS',Name='TDKd3 W5min32 On Not-on six-panel viewing time stack',
                TimeIncrement=.02,TimeIncrementUnit='s',
                Description='Viewing composite only. Six separate cropped fields, not one continuous camera image. '
                    'Native117nm/pixel, display2x; no global composite physical pixel size. '
                    'Classes retained from cached k1 error-ellipse/mask compatibility; identity registration unverified.'))
    assert len(rows)==2500 and sum(r['On'] for r in rows)==57615 and sum(r['Not_on'] for r in rows)==57398
    # tifffile may stop requesting generator items immediately after the last
    # yield, so the final checkpoint belongs after the closed writer.
    cp=dict(rendered_frames=2500,On=57615,Not_on=57398,elapsed_s=time.perf_counter()-started,
        TIFF_bytes_at_checkpoint=target.stat().st_size,TIFF_not_finalized_until_all_frames_complete=False)
    dump(out/'checkpoints/rendered_2500.json',cp);print('CHECKPOINT',json.dumps(cp),flush=True)
    return validate_export(renderer,counts,out)


def validate_export(renderer,counts,out):
    """Read-only validation, also usable after a validation-only interruption."""
    target=out/TIFF_NAME
    saved=pd.read_csv(out/'frame_manifest.csv');assert len(saved)==2500
    np.testing.assert_array_equal(saved.frame,np.arange(2500))
    np.testing.assert_array_equal(saved.page_index,np.arange(2500))
    assert saved.On.sum()==57615 and saved.Not_on.sum()==57398
    rows=saved.to_dict('records')
    np.testing.assert_array_equal(saved.On+saved.Not_on,saved.combined)
    np.testing.assert_array_equal(saved.On,counts.On);np.testing.assert_array_equal(saved.Not_on,counts['Not-on'])
    decoded=[]
    with tifffile.TiffFile(target) as tf:
        assert tf.is_bigtiff and tf.is_ome and len(tf.pages)==2500
        assert tf.series[0].axes=='TYXS' and tf.series[0].shape==(2500,renderer.height,renderer.width,3)
        xml=ET.fromstring(tf.ome_metadata);pixels=xml.find('.//{*}Pixels')
        assert pixels.get('SizeT')=='2500' and pixels.get('SizeZ')=='1' and pixels.get('Type')=='uint8'
        assert float(pixels.get('TimeIncrement'))==.02 and pixels.get('TimeIncrementUnit')=='s'
        for f,page in enumerate(tf.pages):
            image=page.asarray();actual=pixel_hash(image)
            assert actual==rows[f]['rendered_sha256'] and page.dtype==np.dtype('uint8')
            # Series discovery may replace later pages with lightweight
            # TiffFrame instances. Their shared TIFF tags live on keyframe.
            assert page.keyframe.photometric.name=='RGB' and page.keyframe.compression.name in ['DEFLATE','ADOBE_DEFLATE']
            if f in EXAMPLES:
                with Image.open(out/'previews'/f'frame_{f:04d}_six_panels.png') as png:
                    np.testing.assert_array_equal(image,np.asarray(png))
            decoded.append(dict(page_index=f,frame=f,decoded_pixel_sha256=actual,equals_rendered=True))
            if (f+1)%250==0:print('READBACK',f+1,'/2500',flush=True)
        (out/'OME_metadata.xml').write_text(tf.ome_metadata)
    pd.DataFrame(decoded).to_csv(out/'readback_validation.csv',index=False)
    dump(out/'TIFF_validation.json',dict(all2500_pages_readback_equal=True,representative_PNG_pixels_equal=True,
        pages=2500,shape=[2500,renderer.height,renderer.width,3],axes='TYXS',SizeT=2500,SizeZ=1,
        TimeIncrement_s=.02,dtype='uint8',RGB=True,BigTIFF=True,compression='lossless Deflate'))
    if not (out/'checkpoints/rendered_2500.json').exists():
        # Recover a missing post-yield checkpoint only after all pages passed.
        cp=dict(rendered_frames=2500,On=57615,Not_on=57398,elapsed_s=None,
            TIFF_bytes_at_checkpoint=target.stat().st_size,TIFF_not_finalized_until_all_frames_complete=False,
            recovered_after_full_readback=True,
            timing_note='Original generator was not resumed after its final yield; exact rendering completion time was not recorded')
        dump(out/'checkpoints/rendered_2500.json',cp)
    return target


def perform(args,out,c,expected):
    if (out/'parameters.json').exists():assert json.loads((out/'parameters.json').read_text())==c
    else:dump(out/'parameters.json',c)
    dump(out/'run_status.json',dict(status='source_verification'))
    mask,vertices,groups,counts,before=verify(args.run,c,expected,out)
    suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_point_stack.py')
    with (out/'tests.log').open('w') as log:tests=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    assert tests.wasSuccessful(),'Preflight tests failed'
    renderer=Renderer(vertices,c)
    dump(out/'panel_layout.json',dict(canvas_width=renderer.width,canvas_height=renderer.height,
        panel_width=renderer.w,panel_height=renderer.h,panel_origins=renderer.origins,
        coordinate_mapping=c['display_mapping'],native_crop=c['crop_xyxy_px'],scale=2,
        global_physical_pixel_calibration=False))
    dump(out/'versions.json',dict(python=sys.version,executable=sys.executable,
        packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','Pillow','tifffile','h5py']}))
    command=shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--run',str(args.run),
        *(['--prepare-only'] if args.prepare_only else []),*(['--output',str(out)] if args.output else []),
        *(['--validate-existing'] if args.validate_existing else [])])
    command_file='recovery_command.txt' if args.validate_existing else 'preview_command.txt' if args.prepare_only else 'run_command.txt'
    (out/command_file).write_text(command+'\n')
    with tifffile.TiffFile(c['C1_file']) as C1,tifffile.TiffFile(c['C2_file']) as C2:
        previews(renderer,groups,counts,C1,C2,mask,out)
        if args.prepare_only:
            dump(out/'run_status.json',dict(status='previews_ready',representative_frames=EXAMPLES))
            print('PREVIEWS_READY',out,flush=True);return
        target=validate_export(renderer,counts,out) if args.validate_existing else export(renderer,c,groups,counts,C1,C2,mask,out)
    print('Verifying source fingerprints after full export',flush=True)
    after=[fingerprint(p) for p in expected];assert before==after
    dump(out/'input_fingerprints_after.json',after)
    dump(out/'validation.json',dict(all_implementation_checks_passed=True,preflight_tests=tests.testsRun,
        source_fingerprints_unchanged=True,all2500_pages_decoded_and_equal=True,
        input_point_rows_unchanged=True,points=115013,On=57615,Not_on=57398,
        selection_QC_and_ROI_unchanged=True,no_refitting_reclassification_or_TARDIS=True,
        registration_verified=False,viewing_artifact_only=True,final_visual_review_pending=True))
    (out/'README.txt').write_text(
        'TDKd3 W5/min32: six-panel point and raw-image time stack.\n'
        'Panels: On; Not-on; both; raw C2 + circles; raw C1 + full clean-mask contour; raw C1 + full clean-mask contour + circles. Gray outline is the shared cell ROI.\n'
        '2500 original time frames0-2499,20ms/frame; RGB8 OME-BigTIFF, lossless Deflate. T is time, Z=1.\n'
        'This is a viewing artifact with headers, margins and six separate crops, not a scientific intensity or measurement image.\n'
        'Native images remain16-bit,117nm/px. Crop x[82,347),y[232,430),display2x. No global physical calibration for the composite.\n'
        'C2 grayscale406-554ADU, C1 grayscale407-651ADU, fixed for all frames. Native pixels displayed with nearest-neighbor2x replication.\n'
        'Source On57615 and Not-on57398 cached refined centers/QC/ROI/classes retained, no probability thresholding.\n'
        'Point radius1.5native px; circle radius3native px: display sizes only, not uncertainty. Shared floating-center map; overlaps mix colors.\n'
        'Red mask boundaries are closed native foreground pixel-square edges, computed before display or ROI clipping; no artificial ROI cut edges.\n'
        'W5/min32 endpoint frames0,1,2498,2499 use unsmoothed min32 fallback and are labelled.\n'
        'C1/C2 identity registration is unverified; On/Not-on are exploratory spatial compatibility, not binding/cytoplasm proof.\n'
        'frame_manifest.csv records each original frame, counts, source-page and rendered pixel hashes. readback_validation.csv verifies all2500 decoded pages.\n'
        'Source rows, display-coordinate map, parameters/layout, fingerprints, scripts, lock, tests and commands saved.\n'
        'No detection, fitting, segmentation, classification, TARDIS, tracking or significance calculation was run.\n')
    dump(out/'runtime.json',dict(export_file_bytes=target.stat().st_size))
    dump(out/'run_status.json',dict(status='complete_pending_visual_delivery',frames=2500))
    dump(out/'COMPLETED.json',dict(all_implementation_checks_passed=True,final_visual_review_pending=True,
        frames=2500,On=57615,Not_on=57398,TIFF=str(target),output=str(out)))
    (HERE/'latest_point_stack_output.txt').write_text(str(out)+'\n')
    print('COMPLETED',target,flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--validate-existing',action='store_true',help='Validate a fully written existing stack without rendering or writing TIFF data')
    args=parser.parse_args();args.run=args.run.resolve()
    if args.validate_existing and (not args.output or args.prepare_only):parser.error('--validate-existing requires --output and excludes --prepare-only')
    c,expected,base=load_inputs(args.run)
    out=args.output.resolve() if args.output else base/'analysis_results/mito_point_stacks'/(
        'T1_C1C2_W5min32_six_panels_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.output:
        if args.validate_existing:
            assert (out/TIFF_NAME).is_file() and not (out/'COMPLETED.json').exists()
            assert len(pd.read_csv(out/'frame_manifest.csv'))==2500
            archive=out/'source_snapshot/initial_export_render_point_stack.py'
            if not archive.exists():shutil.copyfile(out/'source_snapshot/render_point_stack.py',archive)
            failed=out/'FAILED.json'
            if failed.exists():failed.rename(out/'validation_failure_before_recovery.json')
            dump(out/'validation_recovery.json',dict(TIFF_reused_without_rerender=True,
                reason='TiffFrame tag access corrected to page.keyframe during readback',
                TIFF_before=fingerprint(out/TIFF_NAME)))
        else:
            assert json.loads((out/'run_status.json').read_text())['status']=='previews_ready'
            assert not (out/TIFF_NAME).exists()
    else:
        out.mkdir(parents=True,exist_ok=False)
        for name in ['previews','checkpoints','source_snapshot']:(out/name).mkdir()
    (HERE/'latest_point_stack_attempt.txt').write_text(str(out)+'\n')
    started=time.perf_counter()
    with (out/'run.log').open('a') as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout=Tee(stdout,log);sys.stderr=Tee(stderr,log)
        try:
            print('OUTPUT',out,flush=True);perform(args,out,c,expected)
        except BaseException:
            dump(out/'FAILED.json',dict(traceback=traceback.format_exc(),failed_at_utc=datetime.now(timezone.utc).isoformat()))
            raise
        finally:sys.stdout,sys.stderr=stdout,stderr
    runtime_file='recovery_runtime.json' if args.validate_existing else 'preview_runtime.json' if args.prepare_only else 'runtime.json'
    dump(out/runtime_file,dict(
        elapsed_s=time.perf_counter()-started,prepare_only=args.prepare_only,
        TIFF_bytes=(out/TIFF_NAME).stat().st_size if (out/TIFF_NAME).exists() else None))
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file()
        and p.name not in ['output_fingerprints.json','COMPLETED.json']])


if __name__=='__main__':main()
