#!/usr/bin/env python3
"""Sample-isolated driver for the existing native detector and annular fitter.

Localization depends only on raw C2 and the original cell ROI, so it may run
in parallel with mitochondrial segmentation. No mitochondrial mask is read,
created, verified or claimed to match. The shared detection/fitting/QC code
is reused unchanged. A pilot requires visual review before explicit full run.
"""

import argparse
import ast
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import traceback
import types

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/noton-mplcache')
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import tifffile

import run_sample as shared

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def load_config(path):
    c = json.loads(path.read_text())
    c['config_path'] = str(path.resolve())
    # Preserve the venv entry point instead of resolving its Python symlink.
    c['detector_python'] = str((PROJECT / c['detector_python']).absolute())
    assert c['trackmate_version'] == '7.14.0'
    assert c['radius_um'] == .28 and c['threshold'] == 6.5
    assert c['median_filter'] and c['subpixel_localization']
    assert c['native_detection_chunk_frames'] == 25
    assert c['box_px'] == 7 and c['pixel_size_nm'] == 117 and c['frame_interval_s'] == .02
    assert c['pilot_segments'] == [[0,49],[1225,1274],[2450,2499]]
    assert not c['tracking'] and not c['tardis']
    assert 'mask_directory' not in c
    assert Path(c['sample_id']).name == c['sample_id'] and c['sample_id'] not in ['', '.', '..']
    return c


def validate_roi_sources(c):
    """Read-only header and original ROI/MAT check; no cache directory needed."""
    library, roi_script = shared.roi_reader()
    vertices = library.load_imagej_roi(c['roi_file'])
    np.testing.assert_array_equal(vertices, shared.mat_vertices(c['mat_file'], c['mat_record']))
    assert len(vertices) == c['expected_roi_vertices']
    yy, xx = np.mgrid[:684,:428]
    roi = library.points_in_polygon(np.c_[xx.ravel(),yy.ravel()],vertices).reshape(684,428)
    assert int(roi.sum()) == c['expected_roi_pixels']
    y, x = np.where(roi)
    x0, y0, x1, y1 = c['crop_xyxy_px']
    assert 0 <= x0 <= x.min() and x.max() < x1 <= 428
    assert 0 <= y0 <= y.min() and y.max() < y1 <= 684
    for channel in ['C1_file', 'C2_file']:
        with tifffile.TiffFile(c[channel]) as tf:
            assert len(tf.pages) == 2500 and tf.series[0].axes == 'TYX'
            assert tf.series[0].shape == (2500,684,428) and tf.pages[0].dtype == np.uint16
    report = dict(sample_name=c['sample_name'], sample_id=c['sample_id'],
        ROI_MAT_vertices_exact=True, ROI_pixels=int(roi.sum()), vertices=len(vertices),
        source_shape=[2500,684,428], source_axes='TYX', pixel_size_nm=117,
        frame_interval_s=.02, crop_xyxy_px=list(c['crop_xyxy_px']),
        no_coordinate_transform=True, mitochondrial_mask_required=False,
        mitochondrial_mask_verified=False, selection='Original cell ROI only')
    return roi, vertices, Path(roi_script), report


def prepare(c, out):
    roi, vertices, roi_script, report = validate_roi_sources(c)
    paths = [Path(c[k]) for k in ['C1_file','C2_file','roi_file','mat_file']]
    sources = [Path(__file__).resolve(), HERE/'run_sample.py', HERE/'run_real.py',
        HERE/'run_full.py', HERE/'parallel_fit.py', HERE/'detect_batches.py',
        PROJECT/'trackmate_python_bridge/run_detection.py',
        PROJECT/'trackmate_python_bridge/run_preview.py',
        PROJECT/'localization_overlays/render_frames.py',
        PROJECT/'mito_tardis/render_point_stack.py',PROJECT/'mito_tardis/point_stack_core.py',
        Path(shared.core.base.__file__).resolve(),
        PROJECT/'background_9x9_simulation/local_background/run_local_background.py', roi_script]
    paths += sources + [Path(c['fiji'])/'jars/TrackMate-7.14.0.jar',Path(c['config_path'])]
    before=[]
    for path in paths:
        print('SOURCE',path.name,flush=True);before.append(shared.core.fingerprint(path))
    previous=out/'input_fingerprints_before.json'
    if previous.exists():assert json.loads(previous.read_text()) == before
    shared.core.dump(previous,before)
    pd.DataFrame(vertices,columns=['x_px','y_px']).to_csv(out/'roi_vertices.csv',index=False)
    np.save(out/'roi_mask.npy',roi)
    shared.core.dump(out/'source_validation.json',report)
    for path in sources+[Path(c['config_path'])]:
        relative=path.relative_to(PROJECT)
        target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
    shared.core.dump(out/'rules_before_run.json',c)
    import importlib.metadata
    shared.core.dump(out/'software_versions.json',dict(python=sys.version,executable=sys.executable,
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','tifffile','Pillow','h5py']}))
    freeze=shared.subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (out/'requirements.lock.txt').write_text(freeze.stdout)
    if not (out/'math_validation.json').exists():
        shared.core.dump(out/'math_validation.json',shared.core.math_checks())
    return roi, vertices


def sample_pilot_QC(c,out,fits,groups):
    """Use identical pilot panels, correcting only the legacy sample caption."""
    shared.pilot_QC(c,out,fits,groups)
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',20)
    for frame in [0,1225,2450]:
        path=out/'pilot'/f'frame_{frame:04d}_detection_QC.png'
        with Image.open(path) as source:image=source.copy()
        original_body=np.asarray(image)[70:].copy()
        draw=ImageDraw.Draw(image);draw.rectangle((0,0,image.width,69),fill='black')
        subset=fits[fits.frame==frame]
        draw.text((8,8),f'{c["sample_name"]} | frame {frame} | TM r0.28um + annular7x7 | ROI candidates {len(subset)}',
                  font=font,fill='white')
        draw.text((8,38),f'QC selected {int(subset.qc_unflagged_model_fit.sum())} | nominal model errors, not calibrated coverage',
                  font=font,fill='white')
        np.testing.assert_array_equal(np.asarray(image)[70:],original_body)
        image.save(path,compress_level=6)


def isolated_fit_stage(pointer_directory):
    """Reuse the exact fit-stage code with sample-isolated pointer globals.

    The current shared fit stage uses HERE once, solely to save its legacy
    latest_WT_full_output.txt pointer. Its detector is called separately with
    the original module globals, so no dependency path is redirected. An
    explicit guard fails if a future shared function adds another HERE use.
    The original parallel_fit.fit_batch object remains unchanged for spawn.
    """
    tree=ast.parse(inspect.getsource(shared.fit_stage))
    assert sum(isinstance(n,ast.Name) and n.id=='HERE' for n in ast.walk(tree)) == 1
    globals_copy=dict(shared.fit_stage.__globals__)
    globals_copy.update(HERE=pointer_directory,pilot_QC=sample_pilot_QC)
    fn=types.FunctionType(shared.fit_stage.__code__,globals_copy,shared.fit_stage.__name__,
                          shared.fit_stage.__defaults__,shared.fit_stage.__closure__)
    assert fn.__code__ is shared.fit_stage.__code__
    assert fn.__globals__['fit_batch'] is shared.fit_batch
    assert shared.detect_stage.__globals__['HERE'] == HERE
    return fn


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--stage',choices=['detect-pilot','pilot','full'],default='pilot')
    parser.add_argument('--preflight-only',action='store_true',help='Read TIFF headers and ROI/MAT only; create no cache or pointer')
    args=parser.parse_args();c=load_config(args.config)
    if args.preflight_only:
        _,_,_,report=validate_roi_sources(c);print(json.dumps(report,indent=2));return
    out=args.output.resolve() if args.output else Path(c['C2_file']).parent/'analysis_results/annular_background_refinement'/(
        c['sample_id']+'_C2_r0p28um_box7_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if out.exists():assert json.loads((out/'config.json').read_text())==c
    else:
        out.mkdir(parents=True,exist_ok=False)
        for name in ['source_snapshot','native_detection','pilot','frame_fits','checkpoints']:(out/name).mkdir()
        shared.core.dump(out/'config.json',c)
    (HERE/('latest_'+c['sample_id']+'_attempt.txt')).write_text(str(out)+'\n')
    pointer_directory=HERE/'sample_pointers'/c['sample_id'];pointer_directory.mkdir(parents=True,exist_ok=True)
    fit_stage=isolated_fit_stage(pointer_directory)
    shared.core.dump(out/'orchestration_validation.json',dict(
        shared_fit_stage_code_object_identical=True,shared_native_detector_unchanged=True,
        parallel_fit_batch_object_unchanged=True,fit_stage_global_changes=['HERE','pilot_QC'],
        legacy_pointer_directory=str(pointer_directory),ROI_only_no_mitochondrial_mask_dependency=True))
    command=[sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',c['config_path'],
             '--output',str(out),'--stage',args.stage]
    (out/('command_'+args.stage+'.txt')).write_text(shlex.join(command)+'\n')
    with (out/'run.log').open('a') as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout=shared.Tee(stdout,log);sys.stderr=shared.Tee(stderr,log)
        try:
            print('OUTPUT',out,flush=True);roi,vertices=prepare(c,out)
            frames=shared.stage_frames(c,args.stage)
            groups,hashes=shared.detect_stage(c,out,frames)
            if args.stage=='detect-pilot':
                shared.core.dump(out/'run_status.json',dict(status='pilot_detection_ready'))
                print('PILOT_DETECTION_READY',flush=True);return
            fit_stage(c,out,frames,groups,hashes,roi,vertices,args.stage)
            if args.stage=='full':
                # The legacy source adapter verified its cell-ROI raster
                # against a segmentation directory. This ROI-only driver
                # must not inherit a claim to have verified such an input.
                validation_path=out/'validation.json'
                validation=json.loads(validation_path.read_text())
                validation.pop('ROI_MAT_mask_match',None)
                validation.update(ROI_MAT_vertices_exact=True,
                    cell_ROI_raster_from_original_polygon=True,
                    mitochondrial_mask_required=False,
                    mitochondrial_mask_used_for_localization=False,
                    mitochondrial_mask_verified=False)
                shared.core.dump(validation_path,validation)
                (HERE/('latest_'+c['sample_id']+'_full_output.txt')).write_text(str(out)+'\n')
            else:
                (HERE/('latest_'+c['sample_id']+'_pilot_output.txt')).write_text(str(out)+'\n')
            shared.core.dump(out/'output_fingerprints.json',[shared.core.fingerprint(path) for path in sorted(out.rglob('*'))
                if path.is_file() and path.name not in ['output_fingerprints.json','COMPLETED.json','run.log']])
        except BaseException:
            shared.core.dump(out/'FAILED.json',dict(traceback=traceback.format_exc(),stage=args.stage));raise
        finally:sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':main()
