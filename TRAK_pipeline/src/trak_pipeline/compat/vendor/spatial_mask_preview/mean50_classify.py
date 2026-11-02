"""Classification worker in the existing cleanup environment; no fitting."""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import tifffile

from geometry import classify,point_inside_mask
from run_full import FrameRows,selected,brute_intersection
from run_preview import number,flag,dump
from mean50_contract import frame_to_block,block_metadata

STATUS='exploratory_identity_registration_unverified'
EXTRA=list(block_metadata(0))+['mean50_pointwise_label','mean50_unknown_reason',
    'mean50_center_inside_mask','mean50_intersects_known_mask','mean50_intersects_original_C1_zero_unknown',
    'mean50_min_q','mean50_mask_recipe','mean50_mask_sha256','mean50_registration_status']
REPRESENTATIVE={0,49,50,200,1199,1200,1999,2000,2499}
PREVIEW={200,1200,2000}
LABELS=['mito-compatible','non-mito-compatible','unknown']


def appended(row,mask,zero,c):
    extra=dict(**block_metadata(int(row['frame'])),mean50_pointwise_label='excluded_qc',
        mean50_unknown_reason='',mean50_center_inside_mask=None,mean50_intersects_known_mask=None,
        mean50_intersects_original_C1_zero_unknown=None,mean50_min_q=None,
        mean50_mask_recipe='mean50_min32_holes_lt64',mean50_mask_sha256=c['new_mask_sha256'],
        mean50_registration_status=STATUS)
    if selected(row):
        x,y=number(row,'fit_x_px'),number(row,'fit_y_px')
        a,b=number(row,'x_se_nm')/117,number(row,'y_se_nm')/117
        assert np.isfinite([x,y,a,b]).all() and min(a,b)>0
        if 'pointwise_axis_x_px' in row:
            assert a==number(row,'pointwise_axis_x_px') and b==number(row,'pointwise_axis_y_px')
        result=classify(mask,zero,x,y,a,b,1e-12)
        extra.update(mean50_pointwise_label=result['label'],mean50_unknown_reason=result['reason'],
            mean50_center_inside_mask=point_inside_mask(mask,x,y),
            mean50_intersects_known_mask=result['intersects_known_mask'],
            mean50_intersects_original_C1_zero_unknown=result['intersects_zero_unknown'],mean50_min_q=result['min_q'])
    assert list(extra)==EXTRA
    return {k:('' if v is None else v) for k,v in extra.items()}


def independent(row,extra,mask,zero):
    x,y=number(row,'fit_x_px'),number(row,'fit_y_px')
    a,b=number(row,'x_se_nm')/117,number(row,'y_se_nm')/117
    hit=brute_intersection(mask,x,y,a,b,1e-12);unknown=brute_intersection(zero,x,y,a,b,1e-12)
    height,width=mask.shape
    outside=x-a<-.5-1e-12 or x+a>width-.5+1e-12 or y-b<-.5-1e-12 or y+b>height-.5+1e-12
    label='unknown' if unknown or outside else ('mito-compatible' if hit else 'non-mito-compatible')
    assert label==extra['mean50_pointwise_label'],row['candidate_index']
    assert hit==extra['mean50_intersects_known_mask'] and unknown==extra['mean50_intersects_original_C1_zero_unknown']


def verify_saved(original_path,audit_path,selected_path,c):
    n=keep=0;seen=set()
    with original_path.open(newline='') as a,audit_path.open(newline='') as b,selected_path.open(newline='') as s:
        source=csv.DictReader(a);audit=csv.DictReader(b);chosen=csv.DictReader(s)
        assert audit.fieldnames==source.fieldnames+EXTRA
        for row in source:
            actual=next(audit);assert {k:actual[k] for k in source.fieldnames}==row
            i=int(row['candidate_index']);assert i not in seen;seen.add(i);n+=1
            if selected(row):
                assert next(chosen)==actual;assert actual['mean50_pointwise_label'] in LABELS;keep+=1
            else:assert actual['mean50_pointwise_label']=='excluded_qc'
        assert next(audit,None) is None and next(chosen,None) is None
    assert n==c['expected_ROI_candidates'] and keep==c['expected_selected']
    return dict(all_original_rows_and_cells_preserved_verbatim=True,selected_rows_exactly_preserved=True,
                QC_excluded_not_restored=True,unique_candidate_indices=True,original_rows=n,selected_rows=keep)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resolved',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preview-only',action='store_true')
    args=parser.parse_args();c=json.loads(args.resolved.read_text());out=args.output
    full=np.load(Path(c['mask_directory'])/'clean_full_mask.npy',mmap_mode='r')
    height,width=c.get('image_shape_yx',[684,428])
    assert full.shape==(50,height,width) and full.dtype==bool
    audit=out/('previews/preview_classification.csv' if args.preview_only else 'classification/all_ROI_candidates_mean50.csv')
    chosen=out/('previews/preview_selected.csv' if args.preview_only else 'classification/selected_localizations_mean50.csv')
    frames=[];independent_ids=[];old=Counter();total=Counter();seen=set();start=time.perf_counter()
    with ExitStack() as stack:
        source=FrameRows(stack.enter_context(Path(c['audit_source']).open(newline='')))
        reference=FrameRows(stack.enter_context(Path(c['selected_reference']).open(newline='')))
        assert not set(source.fields)&set(EXTRA)
        streams=[stack.enter_context(p.open('w',newline='')) for p in [audit,chosen]]
        writers=[csv.DictWriter(s,fieldnames=source.fields+EXTRA) for s in streams]
        for w in writers:w.writeheader()
        tf=stack.enter_context(tifffile.TiffFile(c['C1_file']))
        assert len(tf.pages)==2500 and tf.series[0].shape==(2500,height,width)
        for f in range(2500):
            rows=source.for_frame(f);refs={int(r['candidate_index']):r for r in reference.for_frame(f) if selected(r)}
            if args.preview_only and f not in PREVIEW:continue
            selected_rows=[r for r in rows if selected(r)]
            assert {int(r['candidate_index']) for r in selected_rows}==set(refs)
            for row in selected_rows:
                r=refs[int(row['candidate_index'])]
                for key in ['fit_x_px','fit_y_px','x_se_nm','y_se_nm','photons_e','bg_e_per_pixel','sx_px','sy_px']:
                    assert float(row[key])==float(r[key]),(row['candidate_index'],key)
                if 'new_pointwise_label' in row:assert row['new_pointwise_label']==r['new_pointwise_label']
            raw=tf.pages[f].asarray();zero=raw==0;mask=full[frame_to_block(f)]
            counts=Counter({k:0 for k in LABELS});excluded=0
            for row in rows:
                i=int(row['candidate_index']);assert i not in seen;seen.add(i)
                if 'spatial_selected' in row:assert flag(row,'spatial_selected')==selected(row)
                extra=appended(row,mask,zero,c);actual=dict(row,**extra);writers[0].writerow(actual)
                if selected(row):
                    writers[1].writerow(actual);counts[extra['mean50_pointwise_label']]+=1
                    if 'new_pointwise_label' in row:old[row['new_pointwise_label']]+=1
                    if f in REPRESENTATIVE:
                        independent(row,extra,mask,zero);independent_ids.append(i)
                else:excluded+=1
            record=dict(frame=f,time_s=f*.02,block_index=frame_to_block(f),ROI_candidates=len(rows),
                selected=len(selected_rows),QC_excluded=excluded,On=counts[LABELS[0]],Not_on=counts[LABELS[1]],unknown=counts[LABELS[2]])
            assert record['On']+record['Not_on']+record['unknown']==record['selected']
            assert record['selected']+excluded==len(rows)
            frames.append(record);total.update({k:record[k] for k in ['ROI_candidates','selected','QC_excluded','On','Not_on','unknown']})
            if not args.preview_only and (f+1)%250==0:
                for stream in streams:stream.flush()
                pd.DataFrame(frames).to_csv(out/'classification/per_frame_counts.csv',index=False)
                cp=dict(completed_frames=f+1,cumulative_counts=dict(total),elapsed_s=time.perf_counter()-start)
                dump(out/'checkpoints'/f'classified_{f+1:04d}.json',cp)
                print('CLASSIFICATION_CHECKPOINT',json.dumps(cp),flush=True)
        assert source.head is None and reference.head is None
    if args.preview_only:
        if c.get('input_kind')!='annular_cache':
            assert [r['selected'] for r in frames]==[80,45,22]
            assert [r['QC_excluded'] for r in frames]==[13,3,3]
        dump(out/'previews/classification_summary.json',dict(frames=frames,independent_checked_candidates=independent_ids))
    else:
        assert total['selected']==c['expected_selected'] and total['QC_excluded']==c['expected_QC_excluded'] and total['ROI_candidates']==c['expected_ROI_candidates']
        if c.get('input_kind')!='annular_cache':assert [old[k] for k in LABELS]==[57615,57398,0]
        verification=verify_saved(Path(c['audit_source']),audit,chosen,c)
        verification.update(independent_full_pixel_enumeration_passed=True,independent_checked_candidates=independent_ids,
            checked_frames=sorted(REPRESENTATIVE),original_C1_zero_policy_preserved=True,old_W5_counts_reproduced=dict(old))
        dump(out/'classification/validation.json',verification)
        ff=pd.DataFrame(frames);ff.to_csv(out/'classification/per_frame_counts.csv',index=False)
        ff.groupby('block_index')[['ROI_candidates','selected','QC_excluded','On','Not_on','unknown']].sum().reset_index().to_csv(out/'classification/per_block_counts.csv',index=False)
        dump(out/'classification/summary.json',dict(total))
    dump(out/('previews/worker_versions.json' if args.preview_only else 'classification/worker_versions.json'),dict(
        python=sys.version,executable=sys.executable,packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','shapely','tifffile']}))
    print('CLASSIFICATION_COMPLETE',json.dumps(dict(total)),flush=True)


if __name__=='__main__':main()
