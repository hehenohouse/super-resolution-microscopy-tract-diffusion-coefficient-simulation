#!/usr/bin/env python3
"""Classify existing ROI localizations against completed support50 masks; render only pooled dot views."""
import argparse
from collections import Counter
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

import numpy as np
import pandas as pd
import tifffile

HERE=Path(__file__).resolve().parent
LABELS=['mito-compatible','non-mito-compatible','unknown']
REP={0,49,50,200,1199,1200,1999,2000,2499}
EXTRA=['support50_block_index','support50_original_frame_start','support50_original_frame_end_inclusive',
    'support50_pointwise_label','support50_unknown_reason','support50_axis_x_px','support50_axis_y_px',
    'support50_center_inside_mask','support50_mask_sha256','support50_registration_status']
STATUS='exploratory_identity_registration_unverified'


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def worker(c,out):
    from run_full import FrameRows,selected,brute_intersection
    from geometry import classify,point_inside_mask
    full=np.load(Path(c['mask_directory'])/'clean_full_mask.npy',mmap_mode='r')
    mapping=np.load(Path(c['mask_directory'])/'original_frame_to_mask_block.npy')
    source=Path(c['audit_source']);audit=out/'classification/all_ROI_candidates_support50.csv'
    keep=out/c['selected_classification_csv'];frames=[];total=Counter();checked=[];seen=set();start=time.perf_counter()
    with source.open(newline='') as stream,audit.open('w',newline='') as a,keep.open('w',newline='') as k,tifffile.TiffFile(c['C1_file']) as tf:
        rows=FrameRows(stream);fields=rows.fields;assert not set(fields)&set(EXTRA)
        wa=csv.DictWriter(a,fieldnames=fields+EXTRA);wk=csv.DictWriter(k,fieldnames=fields+EXTRA)
        wa.writeheader();wk.writeheader()
        assert len(tf.pages)==2500 and tf.series[0].shape==(2500,684,428)
        for f in range(2500):
            original=rows.for_frame(f);mask=full[mapping[f]];zero=tf.pages[f].asarray()==0;counts=Counter();excluded=0
            for r in original:
                i=int(r['candidate_index']);assert i not in seen;seen.add(i)
                extra=dict(zip(EXTRA,[int(mapping[f]),int(mapping[f])*50,int(mapping[f])*50+49,
                    'excluded_qc','','','','',c['new_mask_sha256'],STATUS]))
                if selected(r):
                    x,y=float(r['fit_x_px']),float(r['fit_y_px']);ax,ay=float(r['x_se_nm'])/117,float(r['y_se_nm'])/117
                    assert np.isfinite([x,y,ax,ay]).all() and min(ax,ay)>0
                    result=classify(mask,zero,x,y,ax,ay,1e-12)
                    extra.update(support50_pointwise_label=result['label'],support50_unknown_reason=result['reason'],
                        support50_axis_x_px=ax,support50_axis_y_px=ay,support50_center_inside_mask=point_inside_mask(mask,x,y))
                    counts[result['label']]+=1
                    if f in REP:
                        hit=brute_intersection(mask,x,y,ax,ay,1e-12);z=brute_intersection(zero,x,y,ax,ay,1e-12)
                        outside=x-ax<-.5-1e-12 or x+ax>427.5+1e-12 or y-ay<-.5-1e-12 or y+ay>683.5+1e-12
                        expected='unknown' if z or outside else ('mito-compatible' if hit else 'non-mito-compatible')
                        assert expected==result['label'] and hit==result['intersects_known_mask'];checked.append(i)
                else:excluded+=1
                actual=dict(r,**extra);wa.writerow(actual)
                if selected(r):wk.writerow(actual)
            record=dict(frame=f,time_s=f*.02,block_index=int(mapping[f]),ROI_candidates=len(original),
                selected=sum(counts.values()),QC_excluded=excluded,On=counts[LABELS[0]],Not_on=counts[LABELS[1]],unknown=counts[LABELS[2]])
            assert record['selected']+excluded==len(original);frames.append(record)
            total.update({key:record[key] for key in ['ROI_candidates','selected','QC_excluded','On','Not_on','unknown']})
            if (f+1)%250==0:
                a.flush();k.flush();dump(out/'checkpoints'/f'classified_{f+1:04d}.json',dict(completed_frames=f+1,counts=dict(total)))
                print('CLASSIFICATION_CHECKPOINT',f+1,json.dumps(dict(total)),flush=True)
        assert rows.head is None
    assert total['selected']==115013 and total['QC_excluded']==12046 and total['ROI_candidates']==127059
    n=kept=0
    with source.open(newline='') as s,audit.open(newline='') as a,keep.open(newline='') as k:
        before=csv.DictReader(s);after=csv.DictReader(a);chosen=csv.DictReader(k)
        assert after.fieldnames==before.fieldnames+EXTRA
        for old in before:
            new=next(after);assert {key:new[key] for key in before.fieldnames}==old;n+=1
            if selected(old):assert next(chosen)==new;kept+=1
            else:assert new['support50_pointwise_label']=='excluded_qc'
        assert next(after,None) is None and next(chosen,None) is None
    assert n==127059 and kept==115013
    frame=pd.DataFrame(frames);frame.to_csv(out/'classification/per_frame_counts.csv',index=False)
    frame.groupby('block_index')[['ROI_candidates','selected','QC_excluded','On','Not_on','unknown']].sum().reset_index().to_csv(out/'classification/per_block_counts.csv',index=False)
    dump(out/'classification/summary.json',dict(total))
    dump(out/'classification/validation.json',dict(all_original_rows_and_cells_preserved_verbatim=True,
        original_QC_and_coordinates_and_SE_unchanged=True,all2500_original_frames_preserved=True,
        counts_conserved=True,independent_full_pixel_enumeration_passed=True,independent_checked_candidates=checked,
        checked_frames=sorted(REP),original_C1_zero_unknown_policy_preserved=True,elapsed_s=time.perf_counter()-start))
    dump(out/'classification/software_versions.json',dict(python=sys.version,executable=sys.executable,
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','shapely','tifffile']}))


def main(args):
    from run_mean50_outputs import project_points,roi_reader,mat_vertices,fingerprint,Tee
    config=json.loads(args.config.read_text());parent=Path(config['localization_view_source'])
    assert json.loads((parent/'COMPLETED.json').read_text())['delivery_checks_passed']
    c=json.loads((parent/'resolved_config.json').read_text());c.update(config)
    c.update(audit_source=str(parent/'classification/all_ROI_candidates_mean50.csv'),previous_projection=str(parent/'projections'),
        selected_classification_csv='classification/selected_localizations_support50.csv',classification_label_field='support50_pointwise_label',
        mask_recipe='support50_ge10_min32_holes_lt64',mask_recipe_display='support50>=10/min32/holes<64',sample_name='TDKd3 T1',
        projection_footer='50-frame support >=10 masks; repeat localization observations retained.',
        output_kind='pooled_localization_views_only',TARDIS=False,refitting=False,resegmentation=False)
    m=Path(c['mask_directory']);base=Path(c['C1_file']).parent
    out=base/'analysis_results/support50_mask_localization_views'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True,exist_ok=False)
    for name in ['classification','projections','source_snapshot','checkpoints']:(out/name).mkdir()
    (HERE/'latest_support50_projection_attempt.txt').write_text(str(out)+'\n')
    command=shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(args.config.resolve())])
    (out/'command.txt').write_text(command+'\n')
    with (out/'run.log').open('a') as log:
        original_out,original_err=sys.stdout,sys.stderr;sys.stdout=Tee(original_out,log);sys.stderr=Tee(original_err,log)
        try:
            print('OUTPUT',out,flush=True)
            known={r['path']:r['sha256'] for r in json.loads((parent/'output_fingerprints.json').read_text())}
            known.update({r['path']:r['sha256'] for r in json.loads((parent/'input_fingerprints_before.json').read_text())})
            known.update({str(m/name):sha for name,sha in json.loads((m/'output_fingerprints.json').read_text()).items()})
            paths=[args.config.resolve(),Path(c['audit_source']),parent/'projections/localization_count_maps.npz',
                parent/'classification/per_frame_counts.csv',Path(c['C1_file']),Path(c['ROI_file']),Path(c['MAT_file']),Path(c['font_path'])]
            paths += [m/name for name in ['clean_full_mask.npy','analysis_ROI_mask.npy','roi_mask.npy','roi_vertices_xy.npy',
                'original_frame_to_mask_block.npy','original_frame_to_mask_block.csv','original_frame_numbers.npy',
                'block_frame_mapping.csv','block_completed.npy','frame_completed.npy','parameters.json','COMPLETE.json',
                'delivery_validation.json','validation.json','foreground_count.npy','input_ge10_full_mask.npy',
                'min32_full_mask.npy','filled_holes_full_mask.npy']]
            before=[]
            for path in paths:
                r=fingerprint(path)
                if str(path) in known:assert r['sha256']==known[str(path)],str(path)
                else:assert path in [args.config.resolve(),Path(c['font_path'])] or path.name=='COMPLETE.json',str(path)
                before.append(r)
            dump(out/'input_fingerprints_before.json',before)
            params=json.loads((m/'parameters.json').read_text());done=json.loads((m/'COMPLETE.json').read_text())
            assert done['all_passed'] and done['source_frames']==2500 and done['mask_blocks']==50
            assert json.loads((m/'delivery_validation.json').read_text())['all_passed']
            assert json.loads((m/'validation.json').read_text())['all_passed']
            assert params['output_shape']==[50,684,428] and params['axes']=='TYX' and params['pixel_size_um']==.117
            assert params['source_frame_interval_s']==.02 and params['mask_block_interval_s']==1.
            assert params['support_minimum_count']==10 and params['support_total_frames']==50
            assert params['minimum_foreground_area_pixel2']==32 and params['foreground_connectivity']==8
            assert params['hole_area_exclusive_max_pixel2']==64 and params['background_connectivity']==4
            assert params['no_image_averaging'] and params['no_additional_sliding_window_voting'] and not params['foreground_boundary_exemptions']
            assert params['source_C1']==c['C1_file']
            for name,shape in [('frame_completed.npy',(2500,)),('block_completed.npy',(50,))]:
                a=np.load(m/name);assert a.shape==shape and a.all()
            mapping=np.load(m/'original_frame_to_mask_block.npy');np.testing.assert_array_equal(mapping,np.arange(2500)//50)
            np.testing.assert_array_equal(np.load(m/'original_frame_numbers.npy'),np.arange(2500))
            fm=pd.read_csv(m/'original_frame_to_mask_block.csv');np.testing.assert_array_equal(fm.original_frame,np.arange(2500))
            np.testing.assert_array_equal(fm.mask_block_index,mapping);np.testing.assert_allclose(fm.timestamp_s,np.arange(2500)*.02,atol=1e-12,rtol=0)
            blocks=pd.read_csv(m/'block_frame_mapping.csv');assert len(blocks)==50
            np.testing.assert_array_equal(blocks.original_frame_start,np.arange(50)*50)
            np.testing.assert_array_equal(blocks.original_frame_end_inclusive,np.arange(50)*50+49)
            np.testing.assert_array_equal(blocks.original_frame_count,np.full(50,50))
            full=np.load(m/'clean_full_mask.npy',mmap_mode='r');roi=np.load(m/'roi_mask.npy')
            assert full.shape==(50,684,428) and full.dtype==bool and roi.shape==(684,428) and roi.dtype==bool and roi.sum()==32179
            np.testing.assert_array_equal(np.load(m/'analysis_ROI_mask.npy'),full&roi)
            counts=np.load(m/'foreground_count.npy',mmap_mode='r');assert counts.shape==full.shape and counts.min()>=0 and counts.max()<=50
            np.testing.assert_array_equal(np.load(m/'input_ge10_full_mask.npy'),counts>=10)
            min32=np.load(m/'min32_full_mask.npy');filled=np.load(m/'filled_holes_full_mask.npy')
            assert not (min32&~full).any();np.testing.assert_array_equal(full,min32|filled);assert not (min32&filled).any()
            lib,roi_path=roi_reader();vertices=lib.load_imagej_roi(c['ROI_file'])
            assert len(vertices)==454;np.testing.assert_array_equal(vertices,np.load(m/'roi_vertices_xy.npy'))
            np.testing.assert_array_equal(vertices,mat_vertices(c['MAT_file'],c['MAT_record']))
            yy,xx=np.mgrid[:684,:428];np.testing.assert_array_equal(roi,lib.points_in_polygon(np.c_[xx.ravel(),yy.ravel()],vertices).reshape(684,428))
            c['new_mask_sha256']=next(r['sha256'] for r in before if r['path']==str(m/'clean_full_mask.npy'))
            dump(out/'resolved_config.json',c);shutil.copyfile(args.config,out/'config.json')
            dump(out/'source_validation.json',dict(all_passed=True,ROI_MAT_mask_vertices_identical=True,
                original_cell_ROI_32179_pixels=True,all2500_frames_to50_blocks_exact=True,recipe_verified=True,
                final_full_mask_and_ROI_intersection_verified=True,registration_status=STATUS))
            test=subprocess.run([c['cleanup_python'],'-B','-m','unittest','discover','-s',str(HERE),'-p','test_geometry.py','-v'],capture_output=True,text=True)
            (out/'geometry_tests.log').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stderr
            worker_command=[c['cleanup_python'],'-B','-u',str(Path(__file__).resolve()),'--worker','--output',str(out)]
            (out/'classification_command.txt').write_text(shlex.join(worker_command)+'\n')
            process=subprocess.Popen(worker_command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:print(line,end='',flush=True)
            assert process.wait()==0,'Classification failed; no partial result accepted'
            frame=pd.read_csv(out/'classification/per_frame_counts.csv');reference=pd.read_csv(parent/'classification/per_frame_counts.csv')
            for field in ['frame','ROI_candidates','selected','QC_excluded']:np.testing.assert_array_equal(frame[field],reference[field])
            project_points(c,out,roi,vertices)
            after=[fingerprint(path) for path in paths];assert before==after;dump(out/'input_fingerprints_after.json',after)
            for name in ['run_support50_projection.py','run_mean50_outputs.py','geometry.py','run_full.py','run_preview.py','contracts.py','test_geometry.py']:
                shutil.copyfile(HERE/name,out/'source_snapshot'/name)
            for name in ['point_stack_core.py','render_point_stack.py','render_time_projection.py']:
                shutil.copyfile(HERE.parent/'mito_tardis'/name,out/'source_snapshot'/name)
            shutil.copyfile(roi_path,out/'source_snapshot/roi_reader.py')
            for source,name in [(parent/'picasso_requirements.lock.txt','picasso_requirements.lock.txt'),(parent/'cleanup_requirements.lock.txt','cleanup_requirements.lock.txt')]:
                shutil.copyfile(source,out/name)
            dump(out/'software_versions.json',dict(python=sys.version,executable=sys.executable,packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','Pillow','tifffile','h5py']}))
            summary=json.loads((out/'classification/summary.json').read_text())
            (out/'README.txt').write_text('TDKd3 T1, all original frames 0..2499 pooled. New support50>=10/min32/holes<64 full masks.\n'
                'Broad per-frame masks -> per-frame min32 -> 50-frame support>=10 -> post-support min32 -> holes<64. No image averaging.\n'
                'Original frame f uses block f//50, a 1s support region, not an instantaneous boundary.\n'
                'Same 115013 QC-pass ROI localizations; 12046 QC exclusions retained, all127059 audit rows and old labels unchanged.\n'
                'Per-point k1 axis-aligned ellipse SE_nm/117, closed pixel squares, tangency included, q tolerance1e-12.\n'
                'Unknown uses original single-frame C1 zero pixels and FOV; never merged into Not-on.\n'
                'Black-background dot views: green On, purple Not-on, gray unknown. Full pooled third panel, no deduplication/weighting.\n'
                'Crop [82,232,347,430], display4x, marker radius0.20 original pixels,2um scale bars. Count grids retain exact counts.\n'
                'C1/C2 identity registration remains unverified; nominal model errors are not validated confidence coverage.\n'
                'On/Not-on describes spatial compatibility, not proven binding/cytoplasm. No refitting, segmentation, tracking or TARDIS.\n')
            dump(out/'validation.json',dict(all_implementation_checks_passed=True,source_fingerprints_unchanged=True,
                original_QC_and_fits_unchanged=True,all_valid_count_grid_identical_to_previous=True,
                visual_review_pending=True,no_TARDIS=True))
            dump(out/'COMPLETED.json',dict(all_implementation_checks_passed=True,visual_review_pending=True,counts=summary))
            (HERE/'latest_support50_projection_output.txt').write_text(str(out)+'\n')
            print('COMPLETED',json.dumps(dict(output=str(out),counts=summary)),flush=True)
        except BaseException:
            import traceback;dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
        finally:sys.stdout,sys.stderr=original_out,original_err
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path);ap.add_argument('--worker',action='store_true');ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    if args.worker:worker(json.loads((args.output/'resolved_config.json').read_text()),args.output)
    else:
        assert args.config is not None;main(args)
