"""Read-only full-movie pointwise ellipse/mask classification and descriptive QC.

Run in the existing cleanup environment. YAML/MAT/ROI source verification runs
in the existing science environment. No localization or segmentation backend,
tracking, trajectory analysis, or TARDIS computation is called.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
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
import shapely
import tifffile
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

from contracts import LABELS, checkpoint_ranges, conserved_counts, resolve_frames
from geometry import classify, pixel_square_union, point_inside_mask, roi_boundary_flags, valid_ellipse
from run_preview import HERE, ROOT, STATUS, LABEL_COLORS, dump, fingerprint, flag, number, outline, stats

EXTRA_FIELDS = [
    'spatial_registration_status','coordinate_transform','XY_covariance_assumption',
    'spatial_selected','spatial_exclusion_reason','original_TrackMate_x_px',
    'original_TrackMate_y_px','original_detection_source_uid',
    'conditional_center_inside_original_mask','pointwise_axis_x_px','pointwise_axis_y_px',
    'pointwise_label','pointwise_unknown_reason','pointwise_touches_ROI_boundary',
    'pointwise_crosses_ROI_boundary','pointwise_min_q_in_local_search',
    'pointwise_intersects_C1_zero_unknown','pointwise_intersects_known_mask',
    'validated_spatial_classification','localization_source_sha256','mask_source_sha256']
CURRENT_OUTPUT = None


class FrameRows:
    """Sorted CSV reader retaining empty original frames and verbatim cell strings."""
    def __init__(self, stream):
        self.reader=csv.DictReader(stream);self.fields=list(self.reader.fieldnames)
        self.previous=-1;self.head=self._next()

    def _next(self):
        row=next(self.reader,None)
        if row is not None:
            frame=int(row['frame'])
            if frame<self.previous:raise ValueError('Source CSV is not sorted by original frame')
            self.previous=frame
        return row

    def for_frame(self, frame):
        if self.head is not None and int(self.head['frame'])<frame:
            raise ValueError('A source frame was skipped')
        rows=[]
        while self.head is not None and int(self.head['frame'])==frame:
            rows.append(self.head);self.head=self._next()
        return rows


def selected(row):
    return flag(row,'qc_unflagged_model_fit') and flag(row,'fitted_center_inside_ROI')


def finite_or_none(v):
    return float(v) if np.isfinite(v) else None


def added_fields(row, det, mask, unknown, polygon, c, fit_hash, mask_hash):
    keep=selected(row);x,y=number(row,'fit_x_px'),number(row,'fit_y_px')
    a,b=number(row,'x_se_nm')/c['pixel_size_nm'],number(row,'y_se_nm')/c['pixel_size_nm']
    reason='' if keep else row['audit_status']
    if not keep and not reason:reason='original_QC_or_refined_ROI_selection_failed'
    result=dict(spatial_registration_status=STATUS,coordinate_transform='identity_assumption',
        XY_covariance_assumption=c['covariance_assumption'],spatial_selected=keep,
        spatial_exclusion_reason=reason,original_TrackMate_x_px=number(det,'x_px'),
        original_TrackMate_y_px=number(det,'y_px'),original_detection_source_uid=det['source_detection_uid'],
        conditional_center_inside_original_mask=point_inside_mask(mask,x,y) if np.isfinite([x,y]).all() else None,
        pointwise_axis_x_px=finite_or_none(a) if keep else None,
        pointwise_axis_y_px=finite_or_none(b) if keep else None,
        pointwise_label='excluded_qc',pointwise_unknown_reason='',
        pointwise_touches_ROI_boundary=None,pointwise_crosses_ROI_boundary=None,
        pointwise_min_q_in_local_search=None,pointwise_intersects_C1_zero_unknown=None,
        pointwise_intersects_known_mask=None,
        validated_spatial_classification='unavailable_registration_unverified',
        localization_source_sha256=fit_hash,mask_source_sha256=mask_hash)
    if keep:
        p=classify(mask,unknown,x,y,a,b,c['intersection_q_tolerance'])
        boundary=roi_boundary_flags(polygon,x,y,a,b,c['intersection_q_tolerance'])
        result.update(pointwise_label=p['label'],pointwise_unknown_reason=p['reason'],
            pointwise_touches_ROI_boundary=boundary['touches_ROI_boundary'],
            pointwise_crosses_ROI_boundary=boundary['crosses_ROI_boundary'],
            pointwise_min_q_in_local_search=p['min_q'],pointwise_intersects_C1_zero_unknown=p['intersects_zero_unknown'],
            pointwise_intersects_known_mask=p['intersects_known_mask'])
    assert list(result)==EXTRA_FIELDS
    return result


def brute_intersection(mask,x,y,a,b,tol):
    """Independent full foreground enumeration, with no production search bounds."""
    yy,xx=np.nonzero(mask)
    dx=np.clip(np.abs(xx.astype(float)-x)-.5,0,None)
    dy=np.clip(np.abs(yy.astype(float)-y)-.5,0,None)
    return bool(np.any(np.square(dx/a)+np.square(dy/b)<=1+tol))


def validate_representative(mask,unknown,rows,extras,c):
    n=0
    for row,e in zip(rows,extras):
        if not e['spatial_selected']:continue
        x,y=number(row,'fit_x_px'),number(row,'fit_y_px');a,b=e['pointwise_axis_x_px'],e['pointwise_axis_y_px']
        if not valid_ellipse(x,y,a,b):
            assert e['pointwise_label']=='unknown';continue
        known=brute_intersection(mask,x,y,a,b,c['intersection_q_tolerance'])
        zero=brute_intersection(unknown,x,y,a,b,c['intersection_q_tolerance'])
        assert known==e['pointwise_intersects_known_mask']
        assert zero==e['pointwise_intersects_C1_zero_unknown']
        h,w=mask.shape;tol=c['intersection_q_tolerance']
        outside=x-a<-.5-tol or x+a>w-.5+tol or y-b<-.5-tol or y+b>h-.5+tol
        label='unknown' if outside or zero else ('mito-compatible' if known else 'non-mito-compatible')
        assert label==e['pointwise_label'];n+=1
    return n


def plot_frame(frame,raw,mask,vertices,rows,extras,summary,c,out):
    low,high=c['display_ADU'];x0,y0,x1,y1=c['display_crop_xyxy_px']
    bounds=(x0-.5,y0-.5,x1-.5,y1-.5)
    fig,axes=plt.subplots(1,2,figsize=c['figure_size_inches'])
    fig.subplots_adjust(left=.018,right=.99,top=.84,bottom=.13,wspace=.025)
    closed=np.vstack([vertices,vertices[0]]);g=pixel_square_union(mask)
    for ax in axes:
        ax.imshow(raw,cmap='gray',vmin=low,vmax=high,origin='upper',interpolation='nearest')
        ax.plot(closed[:,0],closed[:,1],color='#FFDD4A',lw=.7)
        outline(ax,g,'#F17870',.6,visible_bounds=bounds)
        ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[3],bounds[1]),aspect='equal');ax.set_axis_off()
    for row,e in zip(rows,extras):
        if not e['spatial_selected']:
            axes[1].plot(e['original_TrackMate_x_px'],e['original_TrackMate_y_px'],marker='x',
                ms=3.5,mew=.8,color='#FFAA40',ls='none');continue
        x,y=number(row,'fit_x_px'),number(row,'fit_y_px');a,b=e['pointwise_axis_x_px'],e['pointwise_axis_y_px']
        color=LABEL_COLORS[e['pointwise_label']]
        if valid_ellipse(x,y,a,b):axes[1].add_patch(Ellipse((x,y),2*a,2*b,fill=False,edgecolor=color,lw=1))
        axes[1].plot(x,y,'.',ms=2.6,color=color)
    axes[0].set_title('C2 raw + original C1 mask outline',fontsize=11)
    axes[1].set_title(f'Pointwise k=1 | compatible {summary["mito-compatible"]} / non {summary["non-mito-compatible"]} / unknown {summary["unknown"]}',fontsize=10.5)
    fig.suptitle(f'T1 C2 | Original frame {frame} | {frame*c["frame_interval_s"]:.2f} s | {summary["selected"]} selected / {summary["ROI_candidates"]} ROI candidates\n'
        'EXPLORATORY: C1/C2 identity registration UNVERIFIED; individual nominal X/Y model SE',fontsize=12)
    handles=[Line2D([],[],marker='o',ls='none',color=LABEL_COLORS[k],label=name) for k,name in
        [('mito-compatible','Compatible'),('non-mito-compatible','Non-compatible'),('unknown','Unknown')]]
    handles += [Line2D([],[],marker='x',ls='none',color='#FFAA40',label='Excluded QC'),
        Line2D([],[],color='#F17870',label='Original C1 mask'),Line2D([],[],color='#FFDD4A',label='Cell ROI')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.055),ncol=6,fontsize=9,frameon=False)
    fig.text(.5,.02,'406–554 ADU; native C2 grid. Ellipses are not validated confidence regions. Compatibility does not establish binding or cytoplasm.',ha='center',fontsize=8.5)
    path=out/'qc_frames'/f'frame_{frame:04d}_pointwise_k1.png'
    fig.savefig(path,dpi=c['figure_dpi'],facecolor='#171A20',metadata={
        'original_frame':str(frame),'registration_status':STATUS,'selected_points':str(summary['selected'])})
    plt.close(fig);return path


def summarize(frame_records,point_records,out,c):
    frames=pd.DataFrame(frame_records);points=pd.DataFrame(point_records)
    records=[];counts=[]
    ranges=[('full',0,2500)]+[(f'frames_{a:04d}_{a+249:04d}',a,a+250) for a in range(0,2500,250)]
    for name,start,stop in ranges:
        ff=frames[(frames.frame>=start)&(frames.frame<stop)]
        pp=points[(points.frame>=start)&(points.frame<stop)]
        row=dict(scope=name,frame_start=start,frame_stop_exclusive=stop,registration_status=STATUS)
        for field in ['ROI_candidates','selected','excluded',*LABELS,'center_inside_mask','touches_ROI_boundary','crosses_ROI_boundary']:
            row[field]=int(ff[field].sum())
        for field in ff.columns:
            if field.startswith('QC_reason__'):row[field]=int(ff[field].sum())
        conserved_counts(row['ROI_candidates'],row['selected'],row['excluded'],{k:row[k] for k in LABELS})
        row['mean_mask_full_area_px2']=float(ff.mask_full_area_px2.mean())
        row['mean_mask_ROI_area_px2']=float(ff.mask_ROI_area_px2.mean())
        counts.append(row)
        for label in ('all_selected',*LABELS):
            group=pp if label=='all_selected' else pp[pp.label==label]
            r=dict(scope=name,frame_start=start,frame_stop_exclusive=stop,label=label,n=len(group),registration_status=STATUS)
            for field in ['x_se_nm','y_se_nm','photons_e','bg_e_per_pixel','sx_px','sy_px']:
                r.update({field+'_'+key:value for key,value in stats(group[field]).items()})
            records.append(r)
    pd.DataFrame(counts).to_csv(out/'full_and_250_frame_counts.csv',index=False)
    pd.DataFrame(records).to_csv(out/'group_uncertainty_and_signal_summary.csv',index=False)
    dump(out/'summary.json',dict(full=counts[0],segments=counts[1:],group_statistics=records,
        unknown_reasons=dict(Counter(v['unknown_reason'] for v in point_records if v['label']=='unknown'))))
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True,layout='constrained')
    t=frames.time_s
    axes[0].plot(t,frames.ROI_candidates,color='#F6F6F6',lw=.8,label='ROI candidates')
    axes[0].plot(t,frames.excluded,color='#FFAA40',lw=.8,label='Excluded QC')
    for label in LABELS:
        axes[0].plot(t,frames[label],color=LABEL_COLORS[label],lw=.8,label=label)
        axes[1].plot(t,frames[label]/frames.selected.replace(0,np.nan),color=LABEL_COLORS[label],lw=.9,label=label)
    axes[0].set_ylabel('Localizations / frame');axes[0].legend(ncol=3,fontsize=8)
    axes[1].set(ylabel='Fraction of selected',ylim=(-.02,1.02))
    axes[2].plot(t,frames.mask_full_area_px2,color='#F17870',label='Uncut full mask')
    axes[2].plot(t,frames.mask_ROI_area_px2,color='#FFDD4A',label='Mask inside cell ROI')
    axes[2].set(xlabel='Original acquisition time (s)',ylabel='Mask area (pixel²)');axes[2].legend(fontsize=9)
    fig.suptitle('Full 2500 frames | Pointwise k=1 | Exploratory identity registration unverified',fontsize=13)
    fig.savefig(out/'counts_fractions_mask_area_over_time.png',dpi=175);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),layout='constrained')
    Nmax=float(np.quantile(points.photons_e,.995))
    histrows=[];overflows=[]
    for label in LABELS:
        pp=points[points.label==label]
        for axis,field in zip(axes,['x_se_nm','y_se_nm','photons_e']):
            edges=np.linspace(0,200,41) if field!='photons_e' else np.linspace(0,Nmax,51)
            values=pp[field].to_numpy();freq,_=np.histogram(values,bins=edges)
            axis.stairs(freq,edges,color=LABEL_COLORS[label],lw=1.5,label=f'{label} (n={len(pp)})')
            overflows.append(dict(label=label,field=field,shown=int(freq.sum()),
                below=int((values<edges[0]).sum()),above=int((values>edges[-1]).sum()),
                display_min=float(edges[0]),display_max=float(edges[-1])))
            for lo,hi,n in zip(edges[:-1],edges[1:],freq):histrows.append(dict(label=label,field=field,low=float(lo),high=float(hi),count=int(n)))
    for ax,title,unit in zip(axes,['X model SE','Y model SE','Fitted total signal'],['nm','nm','Nominal photoelectron equivalents']):
        ax.set(title=title,xlabel=unit,ylabel='Localizations');ax.legend(fontsize=7)
    fig.suptitle('Descriptive group distributions | Linear axes | Nominal calibration, coverage unvalidated',fontsize=12)
    fig.savefig(out/'group_error_signal_histograms.png',dpi=175);plt.close(fig)
    pd.DataFrame(histrows).to_csv(out/'histogram_bin_counts.csv',index=False)
    dump(out/'histogram_display_ranges_and_overflow.json',overflows)
    return counts,records


def validate_output(out,loc,fields,frames,reference,c):
    """Independent streaming audit; raw cells, QC membership, IDs and checkpoints."""
    audit=out/'all_ROI_candidates_with_spatial_classification.csv'
    selected_path=out/'selected_localizations_with_spatial_classification.csv'
    perframe={f:Counter() for f in frames};n=0;regression=0;seen=set()
    with (loc/'all_ROI_target_fits.csv').open(newline='') as src,audit.open(newline='') as dst,selected_path.open(newline='') as sel:
        sr=csv.DictReader(src);dr=csv.DictReader(dst);kr=csv.DictReader(sel)
        assert list(sr.fieldnames)==fields and list(dr.fieldnames)==fields+EXTRA_FIELDS
        for original in sr:
            actual=next(dr);n+=1
            assert {k:actual[k] for k in fields}==original
            i=int(original['candidate_index']);f=int(original['frame']);assert i not in seen;seen.add(i)
            keep=selected(original);assert flag(actual,'spatial_selected')==keep
            perframe[f]['ROI_candidates']+=1
            if keep:
                assert next(kr)==actual
                label=actual['pointwise_label'];assert label in LABELS
                perframe[f]['selected']+=1;perframe[f][label]+=1
            else:
                assert actual['pointwise_label']=='excluded_qc';perframe[f]['excluded']+=1
            assert actual['spatial_registration_status']==STATUS
            if i in reference:
                old=reference[i]
                assert {k:old[k] for k in fields}==original
                for key in ['spatial_selected','pointwise_label','pointwise_unknown_reason',
                        'pointwise_axis_x_px','pointwise_axis_y_px','pointwise_touches_ROI_boundary',
                        'pointwise_crosses_ROI_boundary','pointwise_intersects_C1_zero_unknown']:
                    assert actual[key]==old[key],(i,key,actual[key],old[key])
                if keep:assert actual['conditional_center_inside_original_mask']==old['conditional_center_inside_original_mask']
                regression+=1
        assert next(dr,None) is None and next(kr,None) is None
    assert regression==len(reference)==166
    summaries=pd.read_csv(out/'per_frame_summary.csv')
    np.testing.assert_array_equal(summaries.frame,frames)
    for item in summaries.to_dict('records'):
        expected=perframe[int(item['frame'])]
        for key in ['ROI_candidates','selected','excluded',*LABELS]:assert item[key]==expected[key]
        conserved_counts(item['ROI_candidates'],item['selected'],item['excluded'],{k:item[k] for k in LABELS})
    blocks=checkpoint_ranges(frames,c['checkpoint_frames'])
    checkpoints=sorted((out/'checkpoints').glob('checkpoint_*.json'));assert len(checkpoints)==len(blocks)
    for path,block in zip(checkpoints,blocks):
        cp=json.loads(path.read_text());assert cp['original_frames']==block
        ff=summaries[summaries.frame.isin(block)]
        for key in ['ROI_candidates','selected','excluded',*LABELS]:assert cp['block_counts'][key]==int(ff[key].sum())
        conserved_counts(cp['block_counts']['ROI_candidates'],cp['block_counts']['selected'],cp['block_counts']['excluded'],{k:cp['block_counts'][k] for k in LABELS})
    return dict(original_rows_preserved_verbatim=n,previous_preview_rows_reproduced_by_candidate_id=regression,
        selected_rows_are_exact_audit_subset=True,all_original_QC_members_and_fit_fields_unchanged=True,
        all_original_frames_and_empty_frames_preserved=True,all_10_checkpoints_conserved=True)


def main(args):
    global CURRENT_OUTPUT
    started=time.perf_counter();verify_cmd=[str(ROOT/'picasso_pilot/.venv/bin/python'),str(HERE/'verify_sources.py'),'--config',str(args.config.resolve())]
    child=subprocess.run(verify_cmd,capture_output=True,text=True)
    if child.returncode:raise RuntimeError('Read-only source verification failed:\n'+child.stderr)
    verified=json.loads(child.stdout)
    c=verified['config'];frames=resolve_frames(c);assert frames==list(range(2500))
    assert c['checkpoint_frames']==250 and c['k']==1
    loc=Path(verified['localization_directory']);mask_dir=Path(verified['mask_directory'])
    out=Path(c['data_directory'])/c['output_relative_directory']/('T1_C2_pointwise_k1_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    CURRENT_OUTPUT = out
    for name in ['checkpoints','qc_frames','source_snapshot']:(out/name).mkdir()
    (HERE/'latest_full_attempt.txt').write_text(str(out)+'\n')
    class Tee:
        def __init__(self,*streams):self.streams=streams
        def write(self,value):
            for s in self.streams:s.write(value);s.flush()
        def flush(self):
            for s in self.streams:s.flush()
    logfile=(out/'run.log').open('w');sys.stdout=Tee(sys.stdout,logfile);sys.stderr=Tee(sys.stderr,logfile)
    print('OUTPUT:',out,flush=True)
    dump(out/'run_status.json',dict(status='verifying_sources',completed_frames=0))
    dump(out/'source_verification.json',verified);dump(out/'resolved_config.json',c);shutil.copyfile(args.config,out/'config_full.yaml')
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve())])+'\n')
    (out/'source_verification_command.txt').write_text(shlex.join(verify_cmd)+'\n')
    for p in HERE.glob('*.py'):shutil.copyfile(p,out/'source_snapshot'/p.name)
    shutil.copyfile(HERE/'README_full.txt',out/'README.txt')
    shutil.copyfile(ROOT/'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py',out/'source_snapshot/locbio.py')
    for folder,name in [('mitochondria_mask_pilot/envs/cleanup','cleanup'),('picasso_pilot','science')]:
        lock=ROOT/folder/'requirements.lock.txt'
        if lock.exists():shutil.copyfile(lock,out/(name+'_requirements.lock.txt'))
    dump(out/'versions.json',dict(python=sys.version,executable=sys.executable,
        cleanup_packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','shapely','matplotlib','Pillow','tifffile']},
        verification_environment=verified['verification_environment']))
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    (out/'tests.log').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    print('Geometry and full-range contract tests passed.',flush=True)
    inputs=list(verified['expected_sha256'])
    for directory,names in [(loc,['rules_before_run.json','source_page_manifest.json','COMPLETED.json']),
            (mask_dir,['parameters.json','validation.json','COMPLETE.json','run_status.json'])]:
        inputs.extend(str(directory/name) for name in names)
    before=[]
    for name in inputs:
        print('Checking source SHA256:',Path(name).name,flush=True);fp=fingerprint(name)
        if name in verified['expected_sha256']:assert fp['sha256']==verified['expected_sha256'][name],name
        before.append(fp)
    dump(out/'input_fingerprints_before.json',before)
    fit_hash=verified['expected_sha256'][str(loc/'all_ROI_target_fits.csv')]
    mask_hash=verified['expected_sha256'][str(mask_dir/'clean_full_mask.npy')]
    full=np.load(mask_dir/'clean_full_mask.npy',mmap_mode='r');analysis=np.load(mask_dir/'analysis_ROI_mask.npy',mmap_mode='r')
    roi=np.load(mask_dir/'roi_mask.npy',mmap_mode='r');vertices=np.asarray(verified['roi_vertices'],float);polygon=shapely.Polygon(vertices)
    assert polygon.is_valid and full.shape==analysis.shape==tuple(c['expected_shape'])
    assert full.dtype==analysis.dtype==roi.dtype==bool and roi.shape==full.shape[1:]
    np.testing.assert_array_equal(np.load(mask_dir/'original_frame_numbers.npy',mmap_mode='r'),frames)
    done=np.load(mask_dir/'frame_completed.npy',mmap_mode='r')
    assert done.shape==(2500,) and done.dtype==bool and done.all()
    mapping=pd.read_csv(mask_dir/'page_frame_mapping.csv');np.testing.assert_array_equal(mapping.page,frames)
    np.testing.assert_array_equal(mapping.original_frame,frames);np.testing.assert_allclose(mapping.time_s,np.arange(2500)*.02,rtol=0,atol=1e-12)
    mask_summaries=pd.read_csv(mask_dir/'per_frame_summary.csv').set_index('frame')
    fit_counts=pd.read_csv(loc/'frame_counts.csv').set_index('frame');np.testing.assert_array_equal(fit_counts.index,frames)
    QC_reasons=['active_parameter_bound','unavailable_support','overlap_review','refined_center_outside_ROI','unavailable_covariance']
    pages={int(r['frame']):r['pixel_sha256'] for r in json.loads((loc/'source_page_manifest.json').read_text())}
    reference_path=Path(c['three_frame_reference_directory'])/'all_three_frame_ROI_candidates_with_spatial_preview.csv'
    with reference_path.open(newline='') as s:reference={int(r['candidate_index']):r for r in csv.DictReader(s)}
    frame_records=[];point_records=[];qc_records=[];unknown_records=[];representative_manifest=[]
    total=Counter();block=Counter();detection_uids=set();candidate_ids=set()
    audit_path=out/'all_ROI_candidates_with_spatial_classification.csv'
    chosen_path=out/'selected_localizations_with_spatial_classification.csv'
    with ExitStack() as stack:
        fit_reader=FrameRows(stack.enter_context((loc/'all_ROI_target_fits.csv').open(newline='')))
        det_reader=FrameRows(stack.enter_context((loc/'all_detections.csv').open(newline='')))
        fields=fit_reader.fields;assert not set(fields)&set(EXTRA_FIELDS)
        audit_stream=stack.enter_context(audit_path.open('w',newline=''));chosen_stream=stack.enter_context(chosen_path.open('w',newline=''))
        audit_writer=csv.DictWriter(audit_stream,fieldnames=fields+EXTRA_FIELDS);audit_writer.writeheader()
        chosen_writer=csv.DictWriter(chosen_stream,fieldnames=fields+EXTRA_FIELDS);chosen_writer.writeheader()
        c1tf=stack.enter_context(tifffile.TiffFile(verified['C1_file']));c2tf=stack.enter_context(tifffile.TiffFile(verified['C2_file']))
        for tf in [c1tf,c2tf]:assert tf.series[0].shape==tuple(c['expected_shape']) and tf.series[0].axes=='TYX' and len(tf.pages)==2500
        for frame in frames:
            rows=fit_reader.for_frame(frame);detections=det_reader.for_frame(frame)
            detections_by_id={int(d['candidate_index']):d for d in detections if int(d['roi_inside'])==1}
            assert len(detections)==fit_counts.loc[frame,'FOV_candidates']
            assert len(rows)==len(detections_by_id)==fit_counts.loc[frame,'ROI_candidates']
            assert {int(r['candidate_index']) for r in rows}==set(detections_by_id)
            keep=[r for r in rows if selected(r)]
            assert len(keep)==fit_counts.loc[frame,'unflagged']
            C1=c1tf.pages[frame].asarray();mask=full[frame];unknown=C1==0
            assert C1.shape==mask.shape and C1.dtype==np.uint16
            np.testing.assert_array_equal(analysis[frame],mask&roi)
            area=int(mask.sum());area_roi=int(analysis[frame].sum())
            assert area==mask_summaries.loc[frame,'clean_full_area_pixel2']
            assert area_roi==mask_summaries.loc[frame,'analysis_ROI_area_pixel2']
            extras=[];counts=Counter({k:0 for k in LABELS});exclusions=Counter();unknown_reasons=Counter()
            center=touch=cross=0
            for row in rows:
                i=int(row['candidate_index']);d=detections_by_id[i]
                assert i not in candidate_ids;candidate_ids.add(i)
                uid=d['source_detection_uid'];assert uid not in detection_uids;detection_uids.add(uid)
                assert int(d['frame'])==frame and d['spot_id']==row['spot_id']
                assert math.isclose(number(row,'time_s'),frame*.02,abs_tol=1e-12)
                if selected(row):
                    assert flag(row,'fit_success') and not flag(row,'active_bound') and flag(row,'finite_uncertainty')
                    x,y=number(row,'fit_x_px'),number(row,'fit_y_px')
                    if np.isfinite([x,y]).all():
                        p=shapely.Point(x,y);assert polygon.covers(p) or polygon.distance(p)<=c['ROI_distance_tolerance_px']
                e=added_fields(row,d,mask,unknown,polygon,c,fit_hash,mask_hash);extras.append(e)
                saved=dict(row,**{key:('' if value is None else value) for key,value in e.items()});audit_writer.writerow(saved)
                if e['spatial_selected']:
                    chosen_writer.writerow(saved);counts[e['pointwise_label']]+=1
                    center+=int(bool(e['conditional_center_inside_original_mask']));touch+=int(bool(e['pointwise_touches_ROI_boundary']))
                    cross+=int(bool(e['pointwise_crosses_ROI_boundary']))
                    if e['pointwise_label']=='unknown':unknown_reasons[e['pointwise_unknown_reason']]+=1
                    record=dict(frame=frame,label=e['pointwise_label'],unknown_reason=e['pointwise_unknown_reason'])
                    for field in ['x_se_nm','y_se_nm','photons_e','bg_e_per_pixel','sx_px','sy_px']:record[field]=number(row,field)
                    point_records.append(record)
                else:exclusions[e['spatial_exclusion_reason']]+=1
            excluded=len(rows)-len(keep);conserved_counts(len(rows),len(keep),excluded,dict(counts))
            assert set(exclusions)<=set(QC_reasons)
            for reason in QC_reasons:
                n=exclusions[reason]
                assert n==fit_counts.loc[frame,reason],(frame,reason,n)
                qc_records.append(dict(frame=frame,time_s=frame*.02,reason=reason,count=n))
            for reason,n in unknown_reasons.items():unknown_records.append(dict(frame=frame,time_s=frame*.02,reason=reason,count=n))
            s=dict(frame=frame,time_s=frame*.02,ROI_candidates=len(rows),selected=len(keep),excluded=excluded,**dict(counts),
                center_inside_mask=center,touches_ROI_boundary=touch,crosses_ROI_boundary=cross,
                mask_full_area_px2=area,mask_ROI_area_px2=area_roi,mask_full_area_um2=area*.117**2,
                mask_ROI_area_um2=area_roi*.117**2,C1_zero_pixels_in_ROI=int((unknown&roi).sum()),
                mean_x_se_nm=stats([number(r,'x_se_nm') for r in keep])['mean'],
                mean_y_se_nm=stats([number(r,'y_se_nm') for r in keep])['mean'],
                mean_photons_e=stats([number(r,'photons_e') for r in keep])['mean'],registration_status=STATUS)
            s.update({'QC_reason__'+reason:exclusions[reason] for reason in QC_reasons})
            frame_records.append(s)
            for key in ['ROI_candidates','selected','excluded',*LABELS]:total[key]+=s[key];block[key]+=s[key]
            if frame in c['representative_frames']:
                raw=c2tf.pages[frame].asarray();assert raw.shape==mask.shape and raw.dtype==np.uint16
                assert hashlib.sha256(raw.tobytes()).hexdigest()==pages[frame]
                checks=validate_representative(mask,unknown,rows,extras,c)
                image=plot_frame(frame,raw,mask,vertices,rows,extras,s,c,out)
                representative_manifest.append(dict(frame=frame,selected=len(keep),independent_full_enumeration_points=checks,
                    C1_pixel_sha256=hashlib.sha256(C1.tobytes()).hexdigest(),C2_pixel_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                    mask_pixel_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),image=str(image.relative_to(out))))
            if (frame+1)%c['checkpoint_frames']==0:
                for stream in [audit_stream,chosen_stream]:stream.flush();os.fsync(stream.fileno())
                first=frame+1-c['checkpoint_frames'];original_frames=list(range(first,frame+1))
                pd.DataFrame(frame_records[-250:]).to_csv(out/'checkpoints'/f'frames_{first:04d}_{frame:04d}_counts.csv',index=False)
                pd.DataFrame(frame_records).to_csv(out/'per_frame_summary.csv',index=False)
                checkpoint=dict(original_frames=original_frames,completed_frames=frame+1,block_counts=dict(block),cumulative_counts=dict(total),
                    audit_file=fingerprint(audit_path),selected_file=fingerprint(chosen_path),elapsed_s=time.perf_counter()-started)
                dump(out/'checkpoints'/f'checkpoint_{frame+1:04d}.json',checkpoint)
                dump(out/'run_status.json',dict(status='classifying',completed_frames=frame+1,cumulative_counts=dict(total)))
                print('CHECKPOINT',json.dumps(dict(frames=f'{first}-{frame}',block=dict(block),cumulative=dict(total)),ensure_ascii=False),flush=True)
                block.clear()
        assert fit_reader.head is None and det_reader.head is None
    assert total['ROI_candidates']==c['expected_ROI_candidates'] and total['selected']==c['expected_selected'] and total['excluded']==c['expected_excluded']
    print('Classification complete; validating CSV fidelity and previous point memberships...',flush=True)
    validation=validate_output(out,loc,fields,frames,reference,c)
    pd.DataFrame(qc_records,columns=['frame','time_s','reason','count']).to_csv(out/'per_frame_QC_exclusion_reasons.csv',index=False)
    pd.DataFrame(unknown_records,columns=['frame','time_s','reason','count']).to_csv(out/'per_frame_unknown_reasons.csv',index=False)
    dump(out/'representative_frame_manifest.json',representative_manifest)
    counts,groupstats=summarize(frame_records,point_records,out,c)
    # Re-read all representative TIFF pages after rendering.
    with tifffile.TiffFile(verified['C1_file']) as tf1,tifffile.TiffFile(verified['C2_file']) as tf2:
        for item in representative_manifest:
            f=item['frame'];assert hashlib.sha256(tf1.pages[f].asarray().tobytes()).hexdigest()==item['C1_pixel_sha256']
            assert hashlib.sha256(tf2.pages[f].asarray().tobytes()).hexdigest()==item['C2_pixel_sha256']
    print('Checking protected source fingerprints after classification...',flush=True)
    after=[fingerprint(p) for p in inputs];dump(out/'input_fingerprints_after.json',after);assert before==after
    validation.update(all_checks_passed=True,original_sources_SHA256_and_stat_unchanged=True,
        representative_frames=c['representative_frames'],independent_full_enumeration_points=sum(r['independent_full_enumeration_points'] for r in representative_manifest),
        geometry_and_contract_tests_passed=True,uncut_full_mask_used=True,registration_verified=False,coverage_validated=False,
        no_refitting_or_resegmentation=True,no_TARDIS_or_tracking=True,no_expanded_mask_movie_created=True)
    dump(out/'validation.json',validation)
    dump(out/'field_units_and_rules.json',dict(original_fields='verbatim source CSV cells; no fitted value or QC change',
        positions='floating native C2 pixels; identity C1/C2 assumption unverified',
        x_se_nm='nominal-calibration full observed-Hessian local model X standard error, nm',y_se_nm='same for Y',
        photons_e='fitted total photoelectron-equivalent signal under nominal calibration, not calibrated incident photons',
        pointwise_axes='each own per-axis SE_nm /117, k=1; diagonal approximation, XY covariance unavailable',
        geometry='closed pixel squares, analytic ellipse/square contact including tangency, q<=1+1e-12',
        unknown='invalid error/position, beyond closed FOV, or ellipse contacts original C1 zero-value pixels',
        selection='original qc_unflagged_model_fit AND fitted_center_inside_ROI',
        ROI='center selection only, no expansion, ellipse ROI contact/crossing is marked but not excluded',
        mean_errors='arithmetic diagnostic only; never used for classification, no division by sqrt(n)',
        mask_areas='pixel² and µm²; scale 0.117 µm/pixel',
        time='original zero-based frame *0.02 seconds; empty frames retained',
        excluded_QC='original audit status; exclusions are not all unavailable errors or proven false positives',
        validated_spatial_classification='always unavailable_registration_unverified'))
    fullcount=counts[0]
    report=(f'全 2500 原始帧（0–2499），逐点误差椭圆法 k=1。\n'
        f'ROI 候选 {total["ROI_candidates"]}；入选 {total["selected"]}；原 QC 排除 {total["excluded"]}。\n'
        f'相容 {total["mito-compatible"]}；不相容 {total["non-mito-compatible"]}；unknown {total["unknown"]}。\n'
        'clean_full_mask 未经 ROI 切断；不扩张细胞 ROI。每点采用自己的 X/Y 标准误，不使用逐帧平均值分类。\n'
        'C1/C2 恒等坐标是假设，精细配准未验证：exploratory_identity_registration_unverified。\n'
        '当前误差来自外圈背景约束拟合的完整观测 Hessian，不是原生 Picasso CRLB；名义校准与置信覆盖均未验证。\n'
        '误差椭圆缺少 XY 协方差，采用对角轴对齐近似；k=1 不能称为已校准二维 68% 置信区域。\n'
        'compatible 不证明结合，non-compatible 不确定等于胞质；unknown 独立保存，未并入不相容组。\n'
        '大误差椭圆更容易接触 mask，组间误差与信号摘要仅供检查，不作显著性或因果结论。\n'
        '所有 ROI 候选的原字段逐字保留；原 QC、拟合、误差、阈值、原始帧号与输入 SHA256 未变。\n'
        '帧 200/1200/2000 按 candidate_index 逐行复现之前相容/不相容 45/35、23/22、12/10。\n'
        f'10 个代表帧的全部入选点共 {validation["independent_full_enumeration_points"]} 个，局部搜索与独立完整像素枚举一致。\n'
        '每 250 帧检查点保存表格与已落盘文件的长度/指纹；所有 10 段数量守恒。\n'
        '原始同帧 C2 及原 C1 mask 轮廓的两栏 QC 图：0、49、200、625、1200、1250、1875、2000、2450、2499。\n'
        '只生成分类缓存和描述性 QC；没有生成 2500 张扩张 mask，没有重新定位/分割、轨迹或 TARDIS。\n')
    (out/'REPORT_zh.txt').write_text(report)
    dump(out/'run_status.json',dict(status='complete',completed_frames=2500,cumulative_counts=dict(total)))
    outputs=[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['run.log','output_fingerprints.json','COMPLETED.json']]
    dump(out/'output_fingerprints.json',outputs)
    dump(out/'COMPLETED.json',dict(all_checks_passed=True,completed_frames=2500,counts=dict(total),output=str(out),
        registration_status=STATUS,elapsed_s=time.perf_counter()-started))
    (HERE/'latest_full_output.txt').write_text(str(out)+'\n')
    print('COMPLETED',json.dumps(dict(output=str(out),counts=dict(total),elapsed_s=time.perf_counter()-started)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    arguments=parser.parse_args()
    try:main(arguments)
    except BaseException:
        if CURRENT_OUTPUT is not None:
            output=CURRENT_OUTPUT
            (output/'failure.txt').write_text(traceback.format_exc())
            dump(output/'run_status.json',dict(status='failed',reason='see failure.txt; incomplete frames are not substituted'))
        raise
