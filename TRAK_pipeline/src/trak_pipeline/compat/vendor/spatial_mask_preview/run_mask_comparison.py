"""Same-point old min16/new W5-min32 mask sensitivity, no localization or TARDIS."""
import argparse
from collections import Counter
from contextlib import ExitStack
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
import time
import traceback

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

from contracts import LABELS, conserved_counts, resolve_frames
from comparison_core import LOSS, GAIN, choose_examples, matrix_from_pairs, transition_name
from comparison_plots import representative, examples_page, count_plots
from geometry import classify, point_inside_mask
from run_full import FrameRows, brute_intersection, selected
from run_preview import ROOT, HERE, STATUS, dump, fingerprint, flag, number, stats

plt.style.use('default')
CURRENT_OUTPUT=None
NEW_FIELDS=['new_mask_recipe','new_mask_source_sha256','new_pointwise_label','new_pointwise_unknown_reason',
    'new_conditional_center_inside_mask','new_pointwise_intersects_known_mask',
    'new_pointwise_intersects_C1_zero_unknown','new_pointwise_min_q_in_local_search',
    'new_mask_temporal_applied','new_mask_input_frames','new_mask_frame_status',
    'classification_transition','classification_changed','comparison_registration_status']


def append_classification(row,mask,unknown,window,c,mask_hash):
    keep=selected(row)
    extra=dict(new_mask_recipe='W5_min32',new_mask_source_sha256=mask_hash,
        new_pointwise_label='excluded_qc',new_pointwise_unknown_reason='',
        new_conditional_center_inside_mask=None,new_pointwise_intersects_known_mask=None,
        new_pointwise_intersects_C1_zero_unknown=None,new_pointwise_min_q_in_local_search=None,
        new_mask_temporal_applied=bool(window['temporal_applied']),new_mask_input_frames=window['input_frames'],
        new_mask_frame_status=window['status'],classification_transition='excluded_qc',
        classification_changed=None,comparison_registration_status=STATUS)
    if keep:
        x,y=number(row,'fit_x_px'),number(row,'fit_y_px');a,b=number(row,'x_se_nm')/117,number(row,'y_se_nm')/117
        assert np.isfinite([x,y,a,b]).all() and min(a,b)>0
        assert a==number(row,'pointwise_axis_x_px') and b==number(row,'pointwise_axis_y_px')
        result=classify(mask,unknown,x,y,a,b,c['intersection_q_tolerance'])
        extra.update(new_pointwise_label=result['label'],new_pointwise_unknown_reason=result['reason'],
            new_conditional_center_inside_mask=point_inside_mask(mask,x,y),
            new_pointwise_intersects_known_mask=result['intersects_known_mask'],
            new_pointwise_intersects_C1_zero_unknown=result['intersects_zero_unknown'],
            new_pointwise_min_q_in_local_search=result['min_q'],
            classification_transition=transition_name(row['pointwise_label'],result['label']),
            classification_changed=row['pointwise_label']!=result['label'])
    assert list(extra)==NEW_FIELDS
    return extra


def independent_check(row,C1,oldmask,newmask,c):
    x,y=number(row,'fit_x_px'),number(row,'fit_y_px');a,b=number(row,'x_se_nm')/117,number(row,'y_se_nm')/117
    zero=brute_intersection(C1==0,x,y,a,b,c['intersection_q_tolerance'])
    outside=x-a<-.5-1e-12 or x+a>427.5+1e-12 or y-b<-.5-1e-12 or y+b>683.5+1e-12
    for mask,label_key in [(oldmask,'pointwise_label'),(newmask,'new_pointwise_label')]:
        hit=brute_intersection(mask,x,y,a,b,c['intersection_q_tolerance'])
        label='unknown' if zero or outside else ('mito-compatible' if hit else 'non-mito-compatible')
        assert label==row[label_key],(row['candidate_index'],label_key,label,row[label_key])
    return True


def validate_tables(out,old_dir,fields,c):
    n=chosen_n=flipped_n=0;matrix=np.zeros((3,3),dtype=np.int64);per_frame={f:Counter() for f in range(2500)}
    seen=set()
    with ExitStack() as stack:
        old=csv.DictReader(stack.enter_context((old_dir/'all_ROI_candidates_with_spatial_classification.csv').open(newline='')))
        audit=csv.DictReader(stack.enter_context((out/'all_ROI_candidates_mask_comparison.csv').open(newline='')))
        prior_selected=csv.DictReader(stack.enter_context((old_dir/'selected_localizations_with_spatial_classification.csv').open(newline='')))
        chosen=csv.DictReader(stack.enter_context((out/'selected_points_mask_comparison.csv').open(newline='')))
        flips=csv.DictReader(stack.enter_context((out/'changed_labels.csv').open(newline='')))
        assert list(old.fieldnames)==fields and list(audit.fieldnames)==fields+NEW_FIELDS
        for original in old:
            actual=next(audit);assert {k:actual[k] for k in fields}==original;n+=1
            i=int(original['candidate_index']);assert i not in seen;seen.add(i)
            f=int(original['frame']);keep=selected(original);assert flag(original,'spatial_selected')==keep
            per_frame[f]['ROI_candidates']+=1
            if keep:
                assert next(prior_selected)==original and next(chosen)==actual;chosen_n+=1
                a,b=LABELS.index(original['pointwise_label']),LABELS.index(actual['new_pointwise_label']);matrix[a,b]+=1
                per_frame[f]['selected']+=1
                per_frame[f]['old_'+('on' if a==0 else 'not_on' if a==1 else 'unknown')]+=1
                per_frame[f]['new_'+('on' if b==0 else 'not_on' if b==1 else 'unknown')]+=1
                if a!=b:assert next(flips)==actual;flipped_n+=1
                assert flag(actual,'classification_changed')==(a!=b)
            else:
                assert actual['new_pointwise_label']=='excluded_qc' and actual['classification_changed']==''
                per_frame[f]['QC_excluded']+=1
        for reader in [audit,prior_selected,chosen,flips]:assert next(reader,None) is None
    assert n==c['expected_ROI_candidates'] and chosen_n==c['expected_selected']
    ff=pd.read_csv(out/'per_frame_comparison.csv');np.testing.assert_array_equal(ff.frame,np.arange(2500))
    for r in ff.to_dict('records'):
        for field in ['ROI_candidates','selected','QC_excluded','old_on','old_not_on','old_unknown','new_on','new_not_on','new_unknown']:
            assert r[field]==per_frame[r['frame']][field]
    checkpoints=sorted((out/'checkpoints').glob('checkpoint_*.json'));assert len(checkpoints)==10
    for p in checkpoints:
        cp=json.loads(p.read_text());block=ff[ff.frame.between(cp['frame_start'],cp['frame_stop_exclusive']-1)]
        for key,value in cp['block_counts'].items():assert int(block[key].sum())==value
    return dict(all_original_127059_rows_and_cells_preserved_verbatim=True,
        selected_115013_points_exactly_match_original_by_ID_and_all_fields=True,
        changed_rows_are_exact_selected_subset=True,paired_matrix=matrix.tolist(),
        changed_points=flipped_n,all_2500_frames_and_10_checkpoints_conserved=True)


def write_summaries(frames,points,matrix,out,c):
    frame_table=pd.DataFrame(frames);frame_table.to_csv(out/'per_frame_comparison.csv',index=False)
    records=[];matrices=[]
    ranges=[('full',0,2500)]+[(f'frames_{a:04d}_{a+249:04d}',a,a+250) for a in range(0,2500,250)]
    for name,lo,hi in ranges:
        ff=frame_table[frame_table.frame.between(lo,hi-1)]
        pp=[p for p in points if lo<=int(p['frame'])<hi]
        mm=matrix_from_pairs((p['old'],p['new']) for p in pp)
        row=dict(scope=name,frame_start=lo,frame_stop_exclusive=hi)
        for key in ['ROI_candidates','selected','QC_excluded','old_on','old_not_on','old_unknown','new_on','new_not_on','new_unknown','on_to_not_on','not_on_to_on','unchanged','changed']:
            row[key]=int(ff[key].sum())
        row['unchanged_fraction']=row['unchanged']/row['selected'] if row['selected'] else None
        row['changed_fraction']=row['changed']/row['selected'] if row['selected'] else None
        records.append(row)
        for i,old in enumerate(LABELS):
            for j,new in enumerate(LABELS):
                n=int(mm[i,j]);den=int(mm[i].sum())
                matrices.append(dict(scope=name,old_label=old,new_label=new,count=n,
                    fraction_of_all_selected=n/len(pp) if pp else None,
                    fraction_of_old_label=n/den if den else None))
    pd.DataFrame(records).to_csv(out/'full_and_250_frame_summary.csv',index=False)
    pd.DataFrame(matrices).to_csv(out/'classification_transition_counts.csv',index=False)
    assert matrix.tolist()==matrix_from_pairs((p['old'],p['new']) for p in points).tolist()
    group_stats=[]
    for old in LABELS:
        for new in LABELS:
            pp=[p for p in points if p['old']==old and p['new']==new]
            record=dict(old_label=old,new_label=new,n=len(pp))
            for field in ['x_se_nm','y_se_nm','photons_e']:
                record.update({field+'_'+k:v for k,v in stats([p[field] for p in pp]).items()})
            group_stats.append(record)
    pd.DataFrame(group_stats).to_csv(out/'transition_error_signal_summary.csv',index=False)
    dump(out/'summary.json',dict(full=records[0],segments=records[1:],matrix_labels=list(LABELS),
        paired_transition_matrix=matrix.tolist(),group_error_signal_statistics=group_stats))
    count_plots(frame_table,matrix,out)
    return records[0]


def main(args):
    global CURRENT_OUTPUT
    start=time.perf_counter();cmd=[str(ROOT/'picasso_pilot/.venv/bin/python'),str(HERE/'verify_comparison_sources.py'),'--config',str(args.config.resolve())]
    child=subprocess.run(cmd,capture_output=True,text=True)
    if child.returncode:raise RuntimeError('Source verification failed:\n'+child.stderr)
    verified=json.loads(child.stdout);c=verified['config'];assert resolve_frames(c)==list(range(2500))
    old_dir=Path(verified['old_classification_directory']);oldmask_dir=Path(verified['old']['mask_directory']);newmask_dir=Path(verified['new']['mask_directory'])
    loc=Path(verified['old']['localization_directory'])
    out=Path(c['data_directory'])/c['output_relative_directory']/('T1_C2_min16_vs_W5min32_k1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False);CURRENT_OUTPUT=out
    for name in ['checkpoints','qc_frames','flip_examples','source_snapshot','source_metadata']:(out/name).mkdir()
    (HERE/'latest_mask_comparison_attempt.txt').write_text(str(out)+'\n')
    class Tee:
        def __init__(self,*streams):self.streams=streams
        def write(self,v):
            for s in self.streams:s.write(v);s.flush()
        def flush(self):
            for s in self.streams:s.flush()
    logfile=(out/'run.log').open('w');sys.stdout=Tee(sys.stdout,logfile);sys.stderr=Tee(sys.stderr,logfile)
    print('OUTPUT',out,flush=True)
    dump(out/'run_status.json',dict(status='verifying_sources',completed_frames=0))
    dump(out/'source_verification.json',verified);dump(out/'resolved_config.json',c);shutil.copyfile(args.config,out/'config_mask_comparison.yaml')
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve())])+'\n')
    (out/'source_verification_command.txt').write_text(shlex.join(cmd)+'\n')
    for p in HERE.glob('*.py'):shutil.copyfile(p,out/'source_snapshot'/p.name)
    shutil.copyfile(HERE/'README_mask_comparison.txt',out/'README.txt')
    shutil.copyfile(ROOT/'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py',out/'source_snapshot/locbio.py')
    for folder,name in [('mitochondria_mask_pilot/envs/cleanup','cleanup'),('picasso_pilot','science')]:
        shutil.copyfile(ROOT/folder/'requirements.lock.txt',out/(name+'_requirements.lock.txt'))
    dump(out/'versions.json',dict(python=sys.version,executable=sys.executable,
        packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','matplotlib','shapely','tifffile','Pillow']},
        scientific_verification=verified['old']['verification_environment']))
    tested=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    (out/'tests.log').write_text(tested.stdout+tested.stderr)
    if tested.returncode:raise RuntimeError(tested.stdout+tested.stderr)
    expected=verified['expected_sha256'];inputs=list(expected)
    for d,names in [(oldmask_dir,['parameters.json','validation.json','COMPLETE.json','run_status.json']),
            (newmask_dir,['parameters.json','validation.json','COMPLETE.json','run_status.json']),
            (old_dir,['COMPLETED.json','resolved_config.json']),
            (loc,['rules_before_run.json','source_page_manifest.json','COMPLETED.json'])]:
        for n in names:
            p=str(d/n)
            if p not in inputs:inputs.append(p)
            shutil.copyfile(d/n,out/'source_metadata'/(d.name+'__'+n))
    before=[]
    for p in inputs:
        print('Checking protected source SHA256:',Path(p).name,flush=True)
        item=fingerprint(p)
        if p in expected:assert item['sha256']==expected[p],p
        before.append(item)
    dump(out/'input_fingerprints_before.json',before)
    oldfull=np.load(oldmask_dir/'clean_full_mask.npy',mmap_mode='r');newfull=np.load(newmask_dir/'clean_full_mask.npy',mmap_mode='r')
    oldROI=np.load(oldmask_dir/'analysis_ROI_mask.npy',mmap_mode='r');newROI=np.load(newmask_dir/'analysis_ROI_mask.npy',mmap_mode='r')
    roi=np.load(newmask_dir/'roi_mask.npy',mmap_mode='r');np.testing.assert_array_equal(roi,np.load(oldmask_dir/'roi_mask.npy',mmap_mode='r'))
    vertices=np.asarray(verified['roi_vertices'],float)
    for d,full,analysis in [(oldmask_dir,oldfull,oldROI),(newmask_dir,newfull,newROI)]:
        assert full.shape==analysis.shape==tuple(c['expected_shape']) and full.dtype==analysis.dtype==bool
        done=np.load(d/'frame_completed.npy',mmap_mode='r');assert done.shape==(2500,) and done.dtype==bool and done.all()
        np.testing.assert_array_equal(np.load(d/'original_frame_numbers.npy',mmap_mode='r'),np.arange(2500))
        mapping=pd.read_csv(d/'page_frame_mapping.csv');np.testing.assert_array_equal(mapping.original_frame,np.arange(2500));np.testing.assert_array_equal(mapping.page,np.arange(2500))
        np.testing.assert_allclose(mapping.time_s,np.arange(2500)*.02,rtol=0,atol=1e-12)
    windows=pd.read_csv(newmask_dir/'frame_windows_and_status.csv',keep_default_na=False).set_index('frame')
    np.testing.assert_array_equal(windows.index,np.arange(2500))
    for f,w in windows.iterrows():
        applied=2<=f<=2497;assert bool(w.temporal_applied)==applied
        assert w.input_frames==','.join(map(str,range(f-2,f+3))) if applied else w.input_frames==str(f)
        assert w.status==('applied' if applied else 'endpoint_unsmoothed_fallback')
        assert abs(w.acquisition_time_s-f*.02)<=1e-12
    newstats=pd.read_csv(newmask_dir/'per_frame_statistics.csv').set_index('frame');oldstats=pd.read_csv(old_dir/'per_frame_summary.csv').set_index('frame')
    sourcepages={int(r['frame']):r['pixel_sha256'] for r in json.loads((loc/'source_page_manifest.json').read_text())}
    new_hash=expected[str(newmask_dir/'clean_full_mask.npy')];old_hash=expected[str(oldmask_dir/'clean_full_mask.npy')]
    frame_records=[];point_records=[];matrix=np.zeros((3,3),dtype=np.int64);manifests=[]
    checks=[];total=Counter();block=Counter();seen=set()
    paths=[out/'all_ROI_candidates_mask_comparison.csv',out/'selected_points_mask_comparison.csv',out/'changed_labels.csv']
    with ExitStack() as stack:
        reader=FrameRows(stack.enter_context((old_dir/'all_ROI_candidates_with_spatial_classification.csv').open(newline='')))
        sourcefit=FrameRows(stack.enter_context((loc/'all_ROI_target_fits.csv').open(newline='')))
        fields=reader.fields;assert not set(fields)&set(NEW_FIELDS)
        streams=[stack.enter_context(p.open('w',newline='')) for p in paths]
        writers=[csv.DictWriter(s,fieldnames=fields+NEW_FIELDS) for s in streams]
        for w in writers:w.writeheader()
        C1tf=stack.enter_context(tifffile.TiffFile(verified['old']['C1_file']));C2tf=stack.enter_context(tifffile.TiffFile(verified['old']['C2_file']))
        for tf in [C1tf,C2tf]:assert tf.series[0].shape==tuple(c['expected_shape']) and tf.series[0].axes=='TYX' and len(tf.pages)==2500
        for f in range(2500):
            rows=reader.for_frame(f);fits=sourcefit.for_frame(f)
            assert [{k:r[k] for k in sourcefit.fields} for r in rows]==fits
            assert len(rows)==oldstats.loc[f,'ROI_candidates']
            C1=C1tf.pages[f].asarray();unknown=C1==0;mask=newfull[f]
            assert C1.shape==mask.shape and C1.dtype==np.uint16
            np.testing.assert_array_equal(newROI[f],mask&roi)
            assert int(mask.sum())==newstats.loc[f,'full_area_pixel2'] and int(newROI[f].sum())==newstats.loc[f,'ROI_area_pixel2']
            pairs=[];merged=[];window=windows.loc[f].to_dict();QC=Counter();counts_old=Counter({k:0 for k in LABELS});counts_new=Counter(counts_old)
            for row in rows:
                i=int(row['candidate_index']);assert i not in seen;seen.add(i)
                assert flag(row,'spatial_selected')==selected(row) and row['mask_source_sha256']==old_hash
                extra=append_classification(row,mask,unknown,window,c,new_hash)
                actual=dict(row,**{k:('' if v is None else v) for k,v in extra.items()})
                writers[0].writerow(actual);merged.append(actual)
                if selected(row):
                    writers[1].writerow(actual);pair=(row['pointwise_label'],extra['new_pointwise_label']);pairs.append(pair)
                    counts_old[pair[0]]+=1;counts_new[pair[1]]+=1
                    if extra['classification_changed']:writers[2].writerow(actual)
                    point_records.append(dict(candidate_index=i,frame=f,old=pair[0],new=pair[1],
                        x_se_nm=number(row,'x_se_nm'),y_se_nm=number(row,'y_se_nm'),photons_e=number(row,'photons_e')))
                else:QC[row['audit_status']]+=1
            mm=matrix_from_pairs(pairs);matrix+=mm;keep=len(pairs);excluded=len(rows)-keep
            assert keep==oldstats.loc[f,'selected'] and excluded==oldstats.loc[f,'excluded']
            for label,n in counts_old.items():assert n==oldstats.loc[f,label]
            conserved_counts(len(rows),keep,excluded,dict(counts_new))
            unchanged=int(np.trace(mm));loss=int(mm[0,1]);gain=int(mm[1,0])
            s=dict(frame=f,time_s=f*.02,ROI_candidates=len(rows),selected=keep,QC_excluded=excluded,
                old_on=counts_old[LABELS[0]],old_not_on=counts_old[LABELS[1]],old_unknown=counts_old[LABELS[2]],
                new_on=counts_new[LABELS[0]],new_not_on=counts_new[LABELS[1]],new_unknown=counts_new[LABELS[2]],
                on_to_not_on=loss,not_on_to_on=gain,unchanged=unchanged,changed=keep-unchanged,
                temporal_applied=window['temporal_applied'],mask_frame_status=window['status'],mask_input_frames=window['input_frames'],
                old_mask_ROI_area_px2=int(oldROI[f].sum()),new_mask_ROI_area_px2=int(newROI[f].sum()),registration_status=STATUS)
            for reason in ['active_parameter_bound','overlap_review','refined_center_outside_ROI','unavailable_support','unavailable_covariance']:
                s['QC_reason__'+reason]=QC[reason]
            for i,old in enumerate(LABELS):
                for j,new in enumerate(LABELS):s['transition__'+old+'__'+new]=int(mm[i,j])
            frame_records.append(s)
            for key in ['ROI_candidates','selected','QC_excluded','new_on','new_not_on','new_unknown','on_to_not_on','not_on_to_on','unchanged','changed']:
                total[key]+=s[key];block[key]+=s[key]
            if f in c['representative_frames']:
                raw=C2tf.pages[f].asarray();assert hashlib.sha256(raw.tobytes()).hexdigest()==sourcepages[f]
                for row in merged:
                    if selected(row):independent_check(row,C1,oldfull[f],mask,c);checks.append(int(row['candidate_index']))
                image=representative(f,C1,raw,oldfull[f],mask,vertices,merged,c,out)
                manifests.append(dict(frame=f,selected=keep,C1_pixel_sha256=hashlib.sha256(C1.tobytes()).hexdigest(),
                    C2_pixel_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),image=str(image.relative_to(out))))
            if (f+1)%250==0:
                for stream in streams:stream.flush();os.fsync(stream.fileno())
                pd.DataFrame(frame_records).to_csv(out/'per_frame_comparison.csv',index=False)
                cp=dict(frame_start=f-249,frame_stop_exclusive=f+1,completed_frames=f+1,block_counts=dict(block),cumulative_counts=dict(total),
                    durable_csv_fingerprints=[fingerprint(p) for p in paths],elapsed_s=time.perf_counter()-start)
                dump(out/'checkpoints'/f'checkpoint_{f+1:04d}.json',cp)
                dump(out/'run_status.json',dict(status='classifying',completed_frames=f+1,cumulative_counts=dict(total)))
                print('CHECKPOINT',json.dumps(dict(frames=f'{f-249}-{f}',block=dict(block),cumulative=dict(total))),flush=True);block.clear()
        assert reader.head is None and sourcefit.head is None
    assert total['ROI_candidates']==127059 and total['selected']==115013 and total['QC_excluded']==12046
    np.testing.assert_array_equal(matrix.sum(axis=1),[c['expected_old_counts'][k] for k in LABELS])
    dump(out/'representative_frame_manifest.json',manifests)
    full_summary=write_summaries(frame_records,point_records,matrix,out,c)
    print('Classification finished; verifying exact source fields and paired subsets...',flush=True)
    validation=validate_tables(out,old_dir,fields,c);assert validation['paired_matrix']==matrix.tolist()
    # Extract examples solely by time/ID order, without selecting signal/quality.
    with (out/'changed_labels.csv').open(newline='') as stream:flips=list(csv.DictReader(stream))
    example_rows={pair:choose_examples([r for r in flips if (r['pointwise_label'],r['new_pointwise_label'])==pair],c['examples_per_direction']) for pair in [LOSS,GAIN]}
    example_manifest=[]
    with tifffile.TiffFile(verified['old']['C1_file']) as tf1,tifffile.TiffFile(verified['old']['C2_file']) as tf2:
        for pair,examples in example_rows.items():
            for row in examples:
                f=int(row['frame']);independent_check(row,tf1.pages[f].asarray(),oldfull[f],newfull[f],c)
                assert hashlib.sha256(tf2.pages[f].asarray().tobytes()).hexdigest()==sourcepages[f]
            image=examples_page(pair,examples,tf1,tf2,oldfull,newfull,vertices,c,out)
            example_manifest.append(dict(direction=transition_name(*pair),candidate_indices=[int(r['candidate_index']) for r in examples],
                frames=[int(r['frame']) for r in examples],image=str(image.relative_to(out)) if image else None))
        for item in manifests:
            f=item['frame'];assert hashlib.sha256(tf1.pages[f].asarray().tobytes()).hexdigest()==item['C1_pixel_sha256']
            assert hashlib.sha256(tf2.pages[f].asarray().tobytes()).hexdigest()==item['C2_pixel_sha256']
    dump(out/'flip_example_manifest.json',example_manifest)
    print('Verifying all protected source fingerprints after comparison...',flush=True)
    after=[fingerprint(p) for p in inputs];dump(out/'input_fingerprints_after.json',after);assert before==after
    validation.update(all_checks_passed=True,geometry_and_recipe_tests_passed=True,
        independent_whole_frame_enumeration_representative_points=len(checks),
        independently_checked_flip_examples=sum(len(v) for v in example_rows.values()),
        protected_input_SHA256_and_stat_unchanged=True,QC_curve_identical_by_original_frame=True,
        identity_registration_verified=False,coverage_validated=False,no_refitting_or_resegmentation=True,no_TARDIS=True)
    dump(out/'validation.json',validation)
    dump(out/'field_units_and_rules.json',dict(original_fields='All original cells unchanged, including old labels and old mask_source_sha256',
        new_labels='Same-point k=1 closed-pixel-square ellipse compatibility under exploratory identity registration',
        x_y_SE='nm, full observed-Hessian nominal-calibration local model SE; not native Picasso CRLB or verified coverage',
        axes='per-point x_se_nm/117,y_se_nm/117, diagonal approximation; no XY covariance available',
        transitions='Different labels for the same localization under different masks, not temporal molecular transfers',
        temporal_mask='W5 3-of-5 majority before full-frame min32; 80 ms center span, 40 ms future context',
        fallback='Frames 0,1,2498,2499 are unsmoothed min32; they can differ from old min16',
        unknown='Same original C1 zero-pixel/FOV policy; never merged with Not-on or QC excluded',
        curves='Raw per-frame counts, no curve smoothing; new mask itself is temporally processed'))
    report=(f'TDKd3 T1；同一批 {total["selected"]} 个入选定位，全部 {total["ROI_candidates"]} 个 ROI 候选原字段保留。\n'
        f'原 QC 排除 {total["QC_excluded"]}，逐帧 QC 曲线不变。\n'
        f'旧 min16：On 59831，Not-on 55182，unknown 0。\n'
        f'新 W5/min32：On {total["new_on"]}，Not-on {total["new_not_on"]}，unknown {total["new_unknown"]}。\n'
        f'保持分类 {total["unchanged"]}；分类变化 {total["changed"]}；On→Not-on {total["on_to_not_on"]}；Not-on→On {total["not_on_to_on"]}。\n'
        '同一原始帧、同一 candidate_index、同一位置/误差/QC，只改变未被 ROI 切断的 mask。\n'
        'C1/C2 恒等坐标假设未独立验证；模型标准误基于名义厂家校准，置信覆盖未验证。\n'
        'On/Not-on 是空间相容性的图例简称，不代表已证明线粒体结合或确定胞质归属。\n'
        '变化同时包含 W5 时间平滑和 min16→min32 面积门槛改动，不能单独归因于其中一个因素。\n'
        '分类翻转是更换 mask 后同一点的不同标签，不是分子跨帧转移，不是转移率。\n'
        '4 个端点帧采用未平滑 min32，可能与旧 min16 不同；原始帧号与空帧保留。\n'
        '10 代表帧的每个入选点与两方向最多6个翻转点用独立完整前景像素枚举验证。\n'
        '图像对应原始同帧页面，C1 407–651 ADU，C2 406–554 ADU，固定显示，不滤波、不平均原图。\n'
        '没有重新定位/分割、没有生成扩张 mask 电影、没有执行 TARDIS/轨迹/运动状态/显著性分析。\n')
    (out/'REPORT_zh.txt').write_text(report)
    dump(out/'run_status.json',dict(status='complete',completed_frames=2500,cumulative_counts=dict(total)))
    dump(out/'output_fingerprints.json',[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['run.log','output_fingerprints.json','COMPLETED.json']])
    dump(out/'COMPLETED.json',dict(all_checks_passed=True,completed_frames=2500,counts=dict(total),paired_matrix=matrix.tolist(),
        output=str(out),registration_status=STATUS,elapsed_s=time.perf_counter()-start))
    (HERE/'latest_mask_comparison_output.txt').write_text(str(out)+'\n')
    print('COMPLETED',json.dumps(dict(output=str(out),counts=dict(total),paired_matrix=matrix.tolist())),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    try:main(args)
    except BaseException:
        if CURRENT_OUTPUT is not None:
            (CURRENT_OUTPUT/'failure.txt').write_text(traceback.format_exc())
            previous=json.loads((CURRENT_OUTPUT/'run_status.json').read_text()) if (CURRENT_OUTPUT/'run_status.json').exists() else {}
            dump(CURRENT_OUTPUT/'run_status.json',dict(previous,status='failed',reason='See failure.txt; no failed frame replaced with empty results'))
        raise
