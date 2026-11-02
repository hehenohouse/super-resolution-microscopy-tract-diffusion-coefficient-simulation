#!/usr/bin/env python3
"""Reclassify frozen localizations against post-support-only min32 masks and plot."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

os.environ.setdefault('MPLCONFIGDIR','/private/tmp/noton-mplcache')
import numpy as np
import pandas as pd
from PIL import Image
import tifffile

HERE=Path(__file__).resolve().parent;PROJECT=HERE.parent
sys.path.insert(0,str(PROJECT/'mito_tardis'))


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

LABELS=['mito-compatible','non-mito-compatible','unknown']
EXTRA=['postmin32_block_index','postmin32_original_frame_start','postmin32_original_frame_end_inclusive',
    'postmin32_pointwise_label','postmin32_unknown_reason','postmin32_axis_x_px','postmin32_axis_y_px',
    'postmin32_center_inside_mask','postmin32_mask_sha256','postmin32_registration_status','postmin32_old_to_new']
STATUS='exploratory_identity_registration_unverified'
REP={0,49,50,200,1199,1200,1999,2000,2499}


def worker(c,out):
    from run_full import FrameRows,selected,brute_intersection
    from geometry import classify,point_inside_mask
    full=np.load(Path(c['mask_directory'])/'clean_full_mask.npy',mmap_mode='r')
    mapping=np.load(Path(c['mask_directory'])/'original_frame_to_mask_block.npy')
    audit=out/'classification/all_ROI_candidates_postmin32.csv';chosen=out/c['selected_classification_csv']
    frames=[];total=Counter();transitions=Counter();checked=[];labels_by_id={};seen=set()
    with Path(c['audit_source']).open(newline='') as stream,audit.open('w',newline='') as a,chosen.open('w',newline='') as k,tifffile.TiffFile(c['C1_file']) as movie:
        rows=FrameRows(stream);fields=rows.fields;assert not set(fields)&set(EXTRA)
        wa=csv.DictWriter(a,fieldnames=fields+EXTRA);wk=csv.DictWriter(k,fieldnames=fields+EXTRA);wa.writeheader();wk.writeheader()
        assert len(movie.pages)==2500 and movie.series[0].shape==(2500,684,428)
        for f in range(2500):
            originals=rows.for_frame(f);b=int(mapping[f]);mask=full[b];zeros=movie.pages[f].asarray()==0;counts=Counter();excluded=0
            for row in originals:
                index=int(row['candidate_index']);assert index not in seen;seen.add(index)
                extra=dict(zip(EXTRA,[b,b*50,b*50+49,'excluded_qc','','','','',c['new_mask_sha256'],STATUS,'excluded_qc']))
                if selected(row):
                    x,y=float(row['fit_x_px']),float(row['fit_y_px']);ax,ay=float(row['x_se_nm'])/117,float(row['y_se_nm'])/117
                    assert np.isfinite([x,y,ax,ay]).all() and min(ax,ay)>0
                    result=classify(mask,zeros,x,y,ax,ay,1e-12);old=row['support50_pointwise_label'];new=result['label']
                    assert old in LABELS and new in LABELS
                    assert (old=='unknown')==(new=='unknown')
                    if old=='mito-compatible':assert new=='mito-compatible'
                    extra.update(postmin32_pointwise_label=new,postmin32_unknown_reason=result['reason'],
                        postmin32_axis_x_px=ax,postmin32_axis_y_px=ay,postmin32_center_inside_mask=point_inside_mask(mask,x,y),
                        postmin32_old_to_new=old+' -> '+new)
                    counts[new]+=1;transitions[(old,new)]+=1;labels_by_id[index]=new
                    if f in REP:
                        hit=brute_intersection(mask,x,y,ax,ay,1e-12);zero=brute_intersection(zeros,x,y,ax,ay,1e-12)
                        outside=x-ax<-.5-1e-12 or x+ax>427.5+1e-12 or y-ay<-.5-1e-12 or y+ay>683.5+1e-12
                        expected='unknown' if zero or outside else ('mito-compatible' if hit else 'non-mito-compatible')
                        assert new==expected and hit==result['intersects_known_mask'];checked.append(index)
                else:excluded+=1
                updated=dict(row,**extra);wa.writerow(updated)
                if selected(row):wk.writerow(updated)
            record=dict(frame=f,time_s=f*.02,block_index=b,ROI_candidates=len(originals),selected=sum(counts.values()),
                QC_excluded=excluded,On=counts[LABELS[0]],Not_on=counts[LABELS[1]],unknown=counts[LABELS[2]])
            assert record['selected']+excluded==len(originals);frames.append(record)
            total.update({key:record[key] for key in ['ROI_candidates','selected','QC_excluded','On','Not_on','unknown']})
            if (f+1)%250==0:
                a.flush();k.flush();dump(out/'checkpoints'/f'classified_{f+1:04d}.json',dict(completed_frames=f+1,counts=dict(total)))
                print('CLASSIFICATION_CHECKPOINT',f+1,json.dumps(dict(total)),flush=True)
        assert rows.head is None
    assert total['selected']==115013 and total['QC_excluded']==12046 and total['ROI_candidates']==127059
    # Verify all old cell strings, including prior classification and calibration, verbatim.
    n=kept=0
    with Path(c['audit_source']).open(newline='') as s,audit.open(newline='') as a,chosen.open(newline='') as k:
        before=csv.DictReader(s);after=csv.DictReader(a);selected_rows=csv.DictReader(k)
        assert after.fieldnames==before.fieldnames+EXTRA
        for old in before:
            new=next(after);assert {key:new[key] for key in before.fieldnames}==old;n+=1
            if selected(old):assert next(selected_rows)==new;kept+=1
            else:assert new['postmin32_pointwise_label']=='excluded_qc'
        assert next(after,None) is None and next(selected_rows,None) is None
    assert n==127059 and kept==115013
    reference=json.loads((PROJECT/c['independent_reference']).read_text())
    digest=hashlib.sha256(''.join(f'{i}\t{labels_by_id[i]}\n' for i in sorted(labels_by_id)).encode()).hexdigest()
    assert digest==reference['classification_ID_sorted_label_lines_sha256']
    for label,key in zip(LABELS,['On','Not_on','unknown']):assert total[key]==reference['classification_counts'][label]
    frame=pd.DataFrame(frames);frame.to_csv(out/'classification/per_frame_counts.csv',index=False)
    frame.groupby('block_index')[['ROI_candidates','selected','QC_excluded','On','Not_on','unknown']].sum().reset_index().to_csv(out/'classification/per_block_counts.csv',index=False)
    pd.DataFrame([dict(old_label=old,new_label=new,count=transitions[(old,new)],fraction_of_selected=transitions[(old,new)]/115013)
        for old in LABELS for new in LABELS]).to_csv(out/'classification/old_new_transition_table.csv',index=False)
    dump(out/'classification/summary.json',dict(total))
    dump(out/'classification/validation.json',dict(all_original_rows_and_cells_preserved_verbatim=True,
        original_QC_coordinates_SE_calibration_and_old_labels_unchanged=True,all2500_original_frames_preserved=True,
        selected_ID_sorted_labels_sha256=digest,independent_all115013_label_results_identical=True,
        independent_full_pixel_enumeration_passed=True,independent_checked_candidates=checked,
        independent_reference_enumeration_count=reference['independent_full_pixel_enumeration_count'],
        no_old_On_became_Not_on=True,original_C1_zero_and_FOV_unknown_policy_preserved=True))
    print('CLASSIFICATION_COMPLETE',json.dumps(dict(total)),flush=True)


def generate(config_path):
    from run_mean50_outputs import project_points
    from render_point_stack import fingerprint,Tee,roi_reader,mat_vertices
    config=json.loads(config_path.read_text());parent=Path(config['localization_view_source'])
    assert json.loads((parent/'COMPLETED.json').read_text())['delivery_checks_passed']
    c=json.loads((parent/'resolved_config.json').read_text());c.update(config)
    c.update(config_path=str(config_path),audit_source=str(parent/'classification/all_ROI_candidates_support50.csv'),
        previous_projection=str(parent/'projections'),selected_classification_csv='classification/selected_localizations_postmin32.csv',
        classification_label_field='postmin32_pointwise_label',mask_recipe='support50_ge10_postmin32_holes_lt64',
        mask_recipe_display='support50>=10/post-min32/holes<64',sample_name='TDKd3 T1',
        projection_footer='50-frame raw-broad support >=10; min32 after support only; repeated observations retained.',
        output_kind='pooled_localization_views_only',TARDIS=False,refitting=False,resegmentation=False,reclassification=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=HERE/'results'/('postmin32_projection_'+stamp);out.mkdir(parents=True,exist_ok=False)
    for name in ['classification','projections','checkpoints','source_snapshot']:(out/name).mkdir()
    c['external_destination']=str(Path(c['C1_file']).parent/'analysis_results/support50_postmin32_localization_views'/stamp)
    (HERE/'latest_postmin32_projection_local.txt').write_text(str(out)+'\n')
    (out/'command.txt').write_text(shlex.join([sys.executable,'-B','-u',str(Path(__file__).resolve()),'--config',str(config_path)])+'\n')
    with (out/'run.log').open('w') as log:
        stdout,stderr=sys.stdout,sys.stderr;sys.stdout=Tee(stdout,log);sys.stderr=Tee(stderr,log)
        try:
            print('OUTPUT',out,flush=True);m=Path(c['mask_directory'])
            assert json.loads((m/'COMPLETE.json').read_text())['all_passed']
            assert json.loads((m/'COMPLETED.json').read_text())['delivery_checks_passed']
            known={r['path']:r['sha256'] for r in json.loads((parent/'output_fingerprints.json').read_text())}
            known.update({r['path']:r['sha256'] for r in json.loads((parent/'input_fingerprints_before.json').read_text())})
            known.update({str(m/name):sha for name,sha in json.loads((m/'output_fingerprints.json').read_text()).items()})
            paths=[config_path,parent/'resolved_config.json',Path(c['audit_source']),parent/'classification/per_frame_counts.csv',parent/'projections/localization_count_maps.npz',
                Path(c['C1_file']),Path(c['ROI_file']),Path(c['MAT_file']),Path(c['font_path']),PROJECT/c['independent_reference']]
            paths.append(Path(json.loads((parent/'resolved_config.json').read_text())['mask_directory'])/'clean_full_mask.npy')
            paths += [m/name for name in ['clean_full_mask.npy','analysis_ROI_mask.npy','roi_mask.npy','roi_vertices_xy.npy',
                'original_frame_to_mask_block.npy','original_frame_numbers_by_block.npy','block_completed.npy','block_frame_mapping.csv',
                'parameters.json','COMPLETE.json','mask_validation.json','validation.json']]
            before=[]
            for path in paths:
                item=fingerprint(path)
                if str(path) in known:assert item['sha256']==known[str(path)],str(path)
                else:assert path in [config_path,Path(c['font_path']),PROJECT/c['independent_reference']] or path.name=='COMPLETE.json',str(path)
                before.append(item)
            dump(out/'input_fingerprints_before.json',before)
            p=json.loads((m/'parameters.json').read_text());assert p['shape']==[50,684,428] and p['axes']=='TYX'
            assert p['pixel_size_um']==.117 and p['source_frame_interval_s']==.02 and p['mask_block_interval_s']==1.
            assert p['per_frame_min32'] is False and p['support_minimum_count']==10 and p['support_total_frames']==50
            assert p['minimum_foreground_area_pixel2']==32 and p['foreground_connectivity']==8
            assert p['hole_area_exclusive_max_pixel2']==64 and p['background_connectivity']==4
            assert p['no_image_averaging'] and p['no_W5_voting'] and not p['foreground_boundary_exemptions']
            assert p['C1_file']==c['C1_file']
            np.testing.assert_array_equal(np.load(m/'original_frame_to_mask_block.npy'),np.arange(2500)//50)
            np.testing.assert_array_equal(np.load(m/'original_frame_numbers_by_block.npy'),np.arange(2500).reshape(50,50))
            assert np.load(m/'block_completed.npy').shape==(50,) and np.load(m/'block_completed.npy').all()
            mapping=pd.read_csv(m/'block_frame_mapping.csv');np.testing.assert_array_equal(mapping.original_frame_start,np.arange(50)*50)
            np.testing.assert_array_equal(mapping.original_frame_end_inclusive,np.arange(50)*50+49);assert mapping.source_frame_count.eq(50).all()
            full=np.load(m/'clean_full_mask.npy',mmap_mode='r');roi=np.load(m/'roi_mask.npy')
            assert full.shape==(50,684,428) and full.dtype==bool and roi.sum()==32179
            np.testing.assert_array_equal(np.load(m/'analysis_ROI_mask.npy'),full&roi)
            oldfull=np.load(Path(json.loads((parent/'resolved_config.json').read_text())['mask_directory'])/'clean_full_mask.npy',mmap_mode='r')
            assert not (oldfull&~full).any()
            lib,roi_path=roi_reader();vertices=lib.load_imagej_roi(c['ROI_file'])
            assert len(vertices)==454;np.testing.assert_array_equal(vertices,np.load(m/'roi_vertices_xy.npy'))
            np.testing.assert_array_equal(vertices,mat_vertices(c['MAT_file'],c['MAT_record']))
            yy,xx=np.mgrid[:684,:428];np.testing.assert_array_equal(roi,lib.points_in_polygon(np.c_[xx.ravel(),yy.ravel()],vertices).reshape(684,428))
            c['new_mask_sha256']=next(r['sha256'] for r in before if r['path']==str(m/'clean_full_mask.npy'))
            dump(out/'resolved_config.json',c);shutil.copyfile(config_path,out/'config.json')
            shutil.copyfile(PROJECT/c['independent_reference'],out/'independent_classification_reference.json')
            dump(out/'source_validation.json',dict(all_passed=True,new_post_support_only_min32_recipe_verified=True,
                ROI_MAT_mask_vertices_identical=True,original_ROI_32179_pixels=True,all2500_frames_to50_blocks_exact=True,
                new_final_contains_old_final=True,registration_status=STATUS))
            test=subprocess.run([c['cleanup_python'],'-B','-m','unittest','discover','-s',str(HERE),'-p','test_geometry.py','-v'],capture_output=True,text=True)
            (out/'geometry_tests.log').write_text(test.stdout+test.stderr);assert test.returncode==0,test.stderr
            command=[c['cleanup_python'],'-B','-u',str(Path(__file__).resolve()),'--worker','--output',str(out)]
            (out/'classification_command.txt').write_text(shlex.join(command)+'\n')
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in child.stdout:print(line,end='',flush=True)
            assert child.wait()==0,'Classification failed; no partial result accepted'
            frame=pd.read_csv(out/'classification/per_frame_counts.csv');old=pd.read_csv(parent/'classification/per_frame_counts.csv')
            for field in ['frame','ROI_candidates','selected','QC_excluded']:np.testing.assert_array_equal(frame[field],old[field])
            project_points(c,out,roi,vertices)
            after=[fingerprint(path) for path in paths];assert before==after;dump(out/'input_fingerprints_after.json',after)
            for name in ['run_postmin32_projection.py','run_mean50_outputs.py','mean50_contract.py','geometry.py','run_full.py','run_preview.py','contracts.py','test_geometry.py']:
                shutil.copyfile(HERE/name,out/'source_snapshot'/name)
            for name in ['point_stack_core.py','render_point_stack.py','render_time_projection.py']:
                shutil.copyfile(PROJECT/'mito_tardis'/name,out/'source_snapshot'/name)
            shutil.copyfile(roi_path,out/'source_snapshot/roi_reader.py')
            dump(out/'software_versions.json',dict(python=sys.version,executable=sys.executable,
                packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','Pillow','tifffile','h5py']}))
            freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True,check=True)
            (out/'requirements.lock.txt').write_text(freeze.stdout)
            (out/'README.txt').write_text('TDKd3 T1 pooled localization views using the new post-support-only min32 mask.\n'
                'Mask: original individual-frame Nellie broad -> per-pixel50-frame support>=10 -> full-image min32 -> enclosed holes<64. No per-frame min32.\n'
                'Reclassified the same115013 QC-pass ROI observations, retaining all127059 audit rows and all previous labels verbatim. QC exclusions12046.\n'
                'Same original centers and nominal-model SEs, k1 diagonal ellipse, SE_nm/117, closed pixel squares, tangency included, q<=1+1e-12.\n'
                'Original frame f uses full mask[f//50]. Unknown retains priority from original single-frame C1 zero pixels and closed FOV; no new registration.\n'
                'Three black-background pooled panels: On green, Not-on purple, all valid combined. Repeat observations retained; no deduplication or error weighting.\n'
                'Same crop[82,232,347,430], display4x, dot radius0.20nativepx and2um scale. Exact all-valid count grid matches old output.\n'
                'C1/C2 identity registration remains unverified; nominal SE is not verified confidence coverage. On/Not-on means spatial compatibility, not proven binding/cytoplasm.\n'
                'No localization fitting, QC change, segmentation, tracking or TARDIS. Mask default flags and paused batch state remain unchanged.\n')
            summary=json.loads((out/'classification/summary.json').read_text())
            dump(out/'validation.json',dict(all_implementation_checks_passed=True,source_fingerprints_unchanged=True,
                original_QC_fits_calibration_old_labels_unchanged=True,all_valid_count_grid_identical=True,
                all_selected_ID_labels_match_independent_reference=True,visual_review_pending=True,no_TARDIS=True))
            dump(out/'COMPLETED.json',dict(all_implementation_checks_passed=True,visual_review_pending=True,output=str(out),counts=summary))
            print('PROJECTIONS_READY',json.dumps(dict(output=str(out),counts=summary)),flush=True)
        except BaseException:
            import traceback;dump(out/'FAILED.json',dict(traceback=traceback.format_exc()));raise
        finally:sys.stdout,sys.stderr=stdout,stderr


def finalize(out):
    from render_point_stack import fingerprint
    c=json.loads((out/'resolved_config.json').read_text());done=json.loads((out/'COMPLETED.json').read_text())
    assert done['all_implementation_checks_passed'] and done['visual_review_pending']
    assert json.loads((out/'input_fingerprints_before.json').read_text())==json.loads((out/'input_fingerprints_after.json').read_text())
    checked=[]
    for name in ['all_timepoints_triptych.png','on_all_timepoints.png','not_on_all_timepoints.png','all_valid_all_timepoints.png']:
        with Image.open(out/'projections'/name) as picture:
            picture.load();checked.append(dict(path=name,pixels=list(picture.size),sha256=fingerprint(out/'projections'/name)['sha256']))
    dump(out/'projections/visual_review.json',dict(passed=True,images=checked,notes='Reviewed pooled triptych and all3individual panels; green/purple/combined point layers, ROI, counts, scale bars and new-recipe labels legible.'))
    counts=done['counts'];assert counts['On']==79615 and counts['Not_on']==35398 and counts['unknown']==0
    assert [json.loads(p.read_text())['completed_frames'] for p in sorted((out/'checkpoints').glob('classified_*.json'))]==list(range(250,2501,250))
    validation=json.loads((out/'validation.json').read_text());validation.update(visual_review_pending=False,visual_review_passed=True)
    dump(out/'validation.json',validation)
    query='import importlib.metadata as m,json,sys;print(json.dumps(dict(python=sys.version,executable=sys.executable,packages={p:m.version(p) for p in ["numpy","pandas","shapely","tifffile","Pillow"]})))'
    versions=subprocess.run([c['cleanup_python'],'-B','-c',query],capture_output=True,text=True,check=True)
    dump(out/'classification/software_versions.json',json.loads(versions.stdout))
    freeze=subprocess.run([c['cleanup_python'],'-m','pip','freeze'],capture_output=True,text=True,check=True)
    (out/'cleanup_requirements.lock.txt').write_text(freeze.stdout)
    done.update(visual_review_pending=False,delivery_checks_passed=True,finished_utc=datetime.now(timezone.utc).isoformat());dump(out/'COMPLETED.json',done)
    shutil.copyfile(Path(__file__),out/'source_snapshot'/Path(__file__).name)
    (out/'finalize_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--finalize','--output',str(out)])+'\n')
    (out/'publish_command.txt').write_text(shlex.join([sys.executable,'-B',str(Path(__file__).resolve()),'--publish','--output',str(out)])+'\n')
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']])
    print('FINALIZED',json.dumps(dict(output=str(out),destination=c['external_destination'],counts=counts)))


def publish(src):
    from render_point_stack import fingerprint
    c=json.loads((src/'resolved_config.json').read_text());dst=Path(c['external_destination'])
    assert dst.parent==Path(c['C1_file']).parent/'analysis_results/support50_postmin32_localization_views' and not dst.exists()
    assert json.loads((src/'COMPLETED.json').read_text())['delivery_checks_passed']
    records=json.loads((src/'output_fingerprints.json').read_text())
    for record in records:
        path=Path(record['path']);assert path.is_relative_to(src) and fingerprint(path)['sha256']==record['sha256']
    dst.mkdir(parents=True)
    for path in sorted(src.rglob('*')):
        target=dst/path.relative_to(src)
        if path.is_dir():target.mkdir(exist_ok=True)
        else:
            assert path.is_file() and not path.is_symlink() and not target.is_symlink();shutil.copyfile(path,target)
    for record in records:
        item=fingerprint(dst/Path(record['path']).relative_to(src));assert item['sha256']==record['sha256'] and item['bytes']==record['bytes']
    dump(dst/'publication.json',dict(local_source=str(src),destination=str(dst),copied_files_SHA256_equal=True,files_verified=len(records)))
    done=json.loads((dst/'COMPLETED.json').read_text());done.update(output=str(dst),local_stage=str(src),external_copy_verified=True);dump(dst/'COMPLETED.json',done)
    dump(dst/'output_fingerprints.json',[fingerprint(p) for p in sorted(dst.rglob('*')) if p.is_file() and p.name not in ['output_fingerprints.json','COMPLETED.json']])
    print('PUBLISHED',json.dumps(dict(output=str(dst),copied_files_SHA256_equal=True,files_verified=len(records))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path);parser.add_argument('--output',type=Path)
    modes=parser.add_mutually_exclusive_group();modes.add_argument('--worker',action='store_true');modes.add_argument('--finalize',action='store_true');modes.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(json.loads((args.output/'resolved_config.json').read_text()),args.output)
    elif args.finalize:finalize(args.output.resolve())
    elif args.publish:publish(args.output.resolve())
    else:
        assert args.config is not None;generate(args.config.resolve())
