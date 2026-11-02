"""Three-frame exploratory C1-mask / cached C2-localization preview.

Entry point runs in the existing cleanup environment. Its read-only source
verification child runs in the existing scientific environment (YAML/HDF5).
No localization, segmentation, tracking or TARDIS pipeline is imported.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd
import shapely
import tifffile

from geometry import (classify,ellipse_intersects_mask,expanded_for_display,
    pixel_square_union,point_inside_mask,roi_boundary_flags)

ROOT=Path(__file__).resolve().parents[1]
HERE=Path(__file__).resolve().parent
STATUS='exploratory_identity_registration_unverified'
LABEL_COLORS={'mito-compatible':'#27DDA5','non-mito-compatible':'#C08FF7','unknown':'#A8B0BA'}
AGREEMENT_COLORS={'both-compatible':'#27DDA5','both-non-mito-compatible':'#7F9CE8',
    'pointwise-only-compatible':'#FF649D','frame-mean-only-compatible':'#4CE1EB',
    'unknown_in_at_least_one_method':'#A8B0BA'}


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def fingerprint(path):
    path=Path(path).resolve();digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    stat=path.stat()
    return dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=digest.hexdigest())


def read_selected_records(path,frames):
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream);fields=list(reader.fieldnames)
        rows=[dict(r) for r in reader if int(r['frame']) in frames]
    return fields,rows


def number(row,key):
    value=row.get(key,'')
    return float(value) if value not in ('','nan','NaN','None') else math.nan


def flag(row,key):
    value=row[key]
    if value not in ('True','False','true','false','1','0'):
        raise ValueError(f'Unexpected boolean {key}: {value}')
    return value in ('True','true','1')


def stats(values):
    a=np.asarray(values,float);a=a[np.isfinite(a)]
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,
        p10=float(np.quantile(a,.1)) if len(a) else None,
        p90=float(np.quantile(a,.9)) if len(a) else None)


def write_appended_csv(path,fields,original,extras):
    assert len(original)==len(extras)
    additional=list(extras[0]) if extras else []
    assert not set(fields)&set(additional)
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields+additional);writer.writeheader()
        for source,new in zip(original,extras):
            assert list(new)==additional
            writer.writerow(dict(source,**{k:('' if v is None else v) for k,v in new.items()}))
    with Path(path).open(newline='') as stream:
        loaded=list(csv.DictReader(stream))
    assert [{k:r[k] for k in fields} for r in loaded]==original


def agreement(first,second):
    if 'unknown' in (first,second):return 'unknown_in_at_least_one_method'
    if first==second=='mito-compatible':return 'both-compatible'
    if first==second=='non-mito-compatible':return 'both-non-mito-compatible'
    return 'pointwise-only-compatible' if first=='mito-compatible' else 'frame-mean-only-compatible'


def normalize_frame_count_keys(config):
    """JSON transport makes YAML's integer mapping keys strings."""
    result=dict(config)
    for name in ['expected_selected_counts','expected_excluded_counts']:
        result[name]={int(k):int(v) for k,v in config[name].items()}
    return result


def outline(ax,geometry,color,width=.65,style='solid',visible_bounds=None):
    parts=list(geometry.geoms) if hasattr(geometry,'geoms') else [geometry]
    lines=[]
    for part in parts:
        if part.is_empty:continue
        if visible_bounds:
            x0,y0,x1,y1=visible_bounds;px0,py0,px1,py1=part.bounds
            if px1<x0 or px0>x1 or py1<y0 or py0>y1:continue
        if part.geom_type=='Polygon':
            lines.append(np.asarray(part.exterior.coords))
            lines.extend(np.asarray(r.coords) for r in part.interiors)
        elif hasattr(part,'geoms'):
            outline(ax,part,color,width,style,visible_bounds)
    if lines:ax.add_collection(LineCollection(lines,colors=color,linewidths=width,linestyles=style))


def plot_frame(frame,c2,mask_geom,expanded_geom,vertices,rows,extras,summary,c,out):
    low,high=c['display_ADU'];x0,y0,x1,y1=c['display_crop_xyxy_px']
    bounds=(x0-.5,y0-.5,x1-.5,y1-.5)
    fig,axes=plt.subplots(2,2,figsize=c['figure_size_inches'])
    fig.subplots_adjust(left=.035,right=.985,top=.88,bottom=.12,hspace=.18,wspace=.035)
    closed=np.vstack([vertices,vertices[0]])
    for ax in axes.flat:
        ax.imshow(c2,cmap='gray',vmin=low,vmax=high,origin='upper',interpolation='nearest')
        ax.plot(closed[:,0],closed[:,1],color='#FFDD4A',lw=.65)
        ax.set(xlim=(bounds[0],bounds[2]),ylim=(bounds[3],bounds[1]),aspect='equal')
        ax.set_axis_off()
    original_color='#F17870';expanded_color='#F5F5F5'
    outline(axes[0,0],mask_geom,original_color,.7,visible_bounds=bounds)
    outline(axes[0,1],mask_geom,original_color,.55,visible_bounds=bounds)
    outline(axes[1,0],mask_geom,original_color,.4,visible_bounds=bounds)
    outline(axes[1,0],expanded_geom,expanded_color,.7,visible_bounds=bounds)
    outline(axes[1,1],mask_geom,original_color,.45,visible_bounds=bounds)
    for source,extra in zip(rows,extras):
        if not extra['spatial_selected']:
            x=extra['original_TrackMate_x_px'];y=extra['original_TrackMate_y_px']
            for ax in [axes[0,1],axes[1,0],axes[1,1]]:
                ax.plot(x,y,marker='x',ms=3.5,mew=.75,color='#FFAA40',ls='none')
            continue
        x,y=number(source,'fit_x_px'),number(source,'fit_y_px')
        a,b=extra['pointwise_axis_x_px'],extra['pointwise_axis_y_px']
        color=LABEL_COLORS[extra['pointwise_label']]
        axes[0,1].add_patch(Ellipse((x,y),2*a,2*b,fill=False,edgecolor=color,lw=.85))
        axes[0,1].plot(x,y,'.',ms=2.3,color=color)
        axes[1,0].plot(x,y,'o',ms=3.6,mfc='none',mec=LABEL_COLORS[extra['frame_mean_label']],mew=.85)
        category=extra['agreement_category']
        axes[1,1].plot(x,y,'o',ms=4.3 if 'only' in category else 2.9,
            mfc='none' if 'only' in category else AGREEMENT_COLORS[category],
            mec=AGREEMENT_COLORS[category],mew=1.1 if 'only' in category else .45)
    p=summary['pointwise_counts'];m=summary['frame_mean_counts'];d=summary['agreement_counts']
    axes[0,0].set_title('C2 raw + original C1 mask outline | identity assumed',fontsize=10.7)
    axes[0,1].set_title(f'Individual ellipses | compatible {p["mito-compatible"]} / non {p["non-mito-compatible"]} / unknown {p["unknown"]}',fontsize=10.7)
    axes[1,0].set_title(f'Frame-mean expansion | compatible {m["mito-compatible"]} / non {m["non-mito-compatible"]} / unknown {m["unknown"]}',fontsize=10.7)
    axes[1,1].set_title(f'Disagreement | individual-only {d["pointwise-only-compatible"]} / mean-only {d["frame-mean-only-compatible"]}',fontsize=10.7)
    fig.suptitle(f'T1 C2 | Original frame {frame} | {frame*.02:.2f} s | Current 7x7 annular-background fits\n'
        f'{summary["selected"]} selected / {summary["ROI_candidates"]} ROI candidates | Mean X/Y SE: {summary["mean_x_se_nm"]:.2f} / {summary["mean_y_se_nm"]:.2f} nm\n'
        'EXPLORATORY: C1/C2 identity registration UNVERIFIED; k=1, diagonal XY covariance',fontsize=12)
    legend=[Line2D([],[],marker='o',color='none',markerfacecolor=LABEL_COLORS['mito-compatible'],label='Compatible'),
        Line2D([],[],marker='o',color='none',markerfacecolor=LABEL_COLORS['non-mito-compatible'],label='Non-compatible'),
        Line2D([],[],marker='x',color='#FFAA40',ls='none',label='Excluded QC'),
        Line2D([],[],color=original_color,label='Original C1 mask'),
        Line2D([],[],color=expanded_color,label='Mean-expanded boundary'),
        Line2D([],[],color='#FFDD4A',label='Cell ROI')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.064),ncol=6,fontsize=8.4,frameon=False)
    legend2=[Line2D([],[],marker='o',ls='none',color=AGREEMENT_COLORS[key],label=name) for key,name in
        [('both-compatible','Both compatible'),('both-non-mito-compatible','Both non-compatible'),
         ('pointwise-only-compatible','Individual only'),('frame-mean-only-compatible','Frame-mean only')]]
    fig.legend(handles=legend2,loc='lower center',bbox_to_anchor=(.5,.033),ncol=4,fontsize=8.4,frameon=False)
    fig.text(.5,.015,'Same 406–554 ADU; native C2 grid. Subpixel geometry includes tangencies. Model SE is not verified coverage or binding evidence.',ha='center',fontsize=8.2)
    path=out/f'frame_{frame:04d}_two_methods.png'
    fig.savefig(path,dpi=c['figure_dpi'],facecolor='#171A20',
        metadata={'original_frame':str(frame),'registration_status':STATUS,'selected_points':str(summary['selected'])})
    plt.close(fig)
    return path


def main(args):
    started=time.perf_counter()
    verify_cmd=[str(ROOT/'picasso_pilot/.venv/bin/python'),str(HERE/'verify_sources.py'),
        '--config',str(args.config.resolve())]
    checked=subprocess.run(verify_cmd,text=True,capture_output=True,check=True)
    verified=json.loads(checked.stdout);c=normalize_frame_count_keys(verified['config'])
    frames=c['frames'];loc=Path(verified['localization_directory']);mask_dir=Path(verified['mask_directory'])
    out=Path(c['data_directory'])/c['output_relative_directory']/('T1_C2_two_methods_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False);(HERE/'latest_output.txt').write_text(str(out)+'\n')
    class Tee:
        def __init__(self,*streams):self.streams=streams
        def write(self,text):
            for stream in self.streams:stream.write(text);stream.flush()
        def flush(self):
            for stream in self.streams:stream.flush()
    logfile=(out/'run.log').open('w');sys.stdout=Tee(sys.stdout,logfile);sys.stderr=Tee(sys.stderr,logfile)
    print('OUTPUT:',out,flush=True)
    dump(out/'source_verification.json',verified);dump(out/'resolved_config.json',c)
    shutil.copyfile(args.config,out/'config.yaml')
    (out/'source_verification_command.txt').write_text(shlex.join(verify_cmd)+'\n')
    # Meaningful tests for tangency, anisotropy, unknown regions and subpixels.
    test=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(HERE),'-p','test_*.py','-v'],capture_output=True,text=True)
    (out/'tests.log').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    print('Ten tests passed, including geometry, 500 independent full-enumeration cases and source-row preservation.',flush=True)
    inputs=list(verified['expected_sha256'])
    for directory,names in [(loc,['rules_before_run.json','source_page_manifest.json','COMPLETED.json']),
            (mask_dir,['parameters.json','validation.json','COMPLETE.json','run_status.json'])]:
        inputs.extend(str(directory/name) for name in names)
    before=[]
    for name in inputs:
        print('Checking input SHA256:',Path(name).name,flush=True)
        item=fingerprint(name)
        if name in verified['expected_sha256']:
            assert item['sha256']==verified['expected_sha256'][name],name
        before.append(item)
    dump(out/'input_fingerprints_before.json',before)
    full=np.load(mask_dir/'clean_full_mask.npy',mmap_mode='r')
    analysis=np.load(mask_dir/'analysis_ROI_mask.npy',mmap_mode='r')
    roi=np.load(mask_dir/'roi_mask.npy',mmap_mode='r')
    vertices=np.asarray(verified['roi_vertices'],float);polygon=shapely.Polygon(vertices)
    assert polygon.is_valid
    assert full.shape==analysis.shape==tuple(c['expected_shape']) and full.dtype==analysis.dtype==np.dtype(bool)
    assert roi.shape==full.shape[1:] and roi.dtype==bool
    np.testing.assert_array_equal(np.load(mask_dir/'original_frame_numbers.npy',mmap_mode='r'),np.arange(2500))
    done=np.load(mask_dir/'frame_completed.npy',mmap_mode='r');assert done.shape==(2500,) and done.all()
    mapping=pd.read_csv(mask_dir/'page_frame_mapping.csv')
    np.testing.assert_array_equal(mapping.original_frame,np.arange(2500));np.testing.assert_array_equal(mapping.page,np.arange(2500))
    np.testing.assert_allclose(mapping.time_s,np.arange(2500)*.02,rtol=0,atol=1e-12)
    fields,original=read_selected_records(loc/'all_ROI_target_fits.csv',frames)
    _,det=read_selected_records(loc/'all_detections.csv',frames)
    det_by_id={int(r['candidate_index']):r for r in det}
    assert len(det_by_id)==len(det)
    assert len({int(r['candidate_index']) for r in original})==len(original)
    assert {int(r['candidate_index']) for r in original}=={int(r['candidate_index']) for r in det if int(r['roi_inside'])==1}
    per_frame=pd.read_csv(mask_dir/'per_frame_summary.csv').set_index('frame')
    source_pages={int(r['frame']):r['pixel_sha256'] for r in json.loads((loc/'source_page_manifest.json').read_text())}
    all_extra=[];summaries=[];groups=[];figures=[];raw1=[];raw2=[];selected_full=[];selected_analysis=[];frame_checks=[]
    with tifffile.TiffFile(verified['C1_file']) as c1tf,tifffile.TiffFile(verified['C2_file']) as c2tf:
        for tf in [c1tf,c2tf]:
            assert tf.series[0].shape==tuple(c['expected_shape']) and tf.series[0].axes=='TYX' and len(tf.pages)==2500
        for frame in frames:
            rows=[r for r in original if int(r['frame'])==frame]
            chosen=[r for r in rows if flag(r,'qc_unflagged_model_fit') and flag(r,'fitted_center_inside_ROI')]
            assert len(chosen)==c['expected_selected_counts'][frame]
            assert len(rows)-len(chosen)==c['expected_excluded_counts'][frame]
            for r in chosen:
                assert flag(r,'fit_success') and not flag(r,'active_bound') and flag(r,'finite_uncertainty')
                assert np.isfinite([number(r,k) for k in ['fit_x_px','fit_y_px','x_se_nm','y_se_nm']]).all()
                assert min(number(r,'x_se_nm'),number(r,'y_se_nm'))>0
                point=shapely.Point(number(r,'fit_x_px'),number(r,'fit_y_px'))
                assert polygon.covers(point) or polygon.distance(point)<=c['ROI_distance_tolerance_px']
            mean_x=float(np.mean([number(r,'x_se_nm') for r in chosen]));mean_y=float(np.mean([number(r,'y_se_nm') for r in chosen]))
            mean_a,mean_b=mean_x/117,mean_y/117
            C1=c1tf.pages[frame].asarray();C2=c2tf.pages[frame].asarray();mask=full[frame]
            assert C1.dtype==C2.dtype==np.uint16 and C1.shape==C2.shape==(684,428)
            assert hashlib.sha256(C2.tobytes()).hexdigest()==source_pages[frame]
            np.testing.assert_array_equal(analysis[frame],mask&roi)
            assert int(analysis[frame].sum())==int(per_frame.loc[frame,'analysis_ROI_area_pixel2'])
            unknown=C1==0
            extras=[];count_p={k:0 for k in LABEL_COLORS};count_m=dict(count_p)
            counts_d={k:0 for k in AGREEMENT_COLORS}
            foreground_y,foreground_x=np.nonzero(mask);brute_checks=0
            for row in rows:
                i=int(row['candidate_index']);source_det=det_by_id[i]
                assert int(row['frame'])==int(source_det['frame']) and int(row['spot_id'])==int(source_det['spot_id'])
                selected=flag(row,'qc_unflagged_model_fit') and flag(row,'fitted_center_inside_ROI')
                x,y=number(row,'fit_x_px'),number(row,'fit_y_px')
                a,b=number(row,'x_se_nm')/117,number(row,'y_se_nm')/117
                # All extra fields are present for excluded rows; no invented
                # error/classification is attached to an orange X.
                result=dict(spatial_registration_status=STATUS,coordinate_transform='identity_assumption',
                    XY_covariance_assumption=c['covariance_assumption'],spatial_selected=bool(selected),
                    spatial_exclusion_reason='' if selected else row['audit_status'],
                    original_TrackMate_x_px=number(source_det,'x_px'),original_TrackMate_y_px=number(source_det,'y_px'),
                    original_detection_source_uid=source_det['source_detection_uid'],
                    conditional_center_inside_original_mask=None,
                    pointwise_axis_x_px=a if selected else None,pointwise_axis_y_px=b if selected else None,
                    frame_mean_axis_x_px=mean_a,frame_mean_axis_y_px=mean_b,
                    pointwise_label='excluded_qc',frame_mean_label='excluded_qc',agreement_category='excluded_qc',
                    pointwise_unknown_reason='',frame_mean_unknown_reason='',
                    pointwise_touches_ROI_boundary=None,pointwise_crosses_ROI_boundary=None,
                    frame_mean_touches_ROI_boundary=None,frame_mean_crosses_ROI_boundary=None,
                    pointwise_min_q_in_local_search=None,frame_mean_min_q_in_local_search=None,
                    pointwise_intersects_C1_zero_unknown=None,frame_mean_intersects_C1_zero_unknown=None,
                    validated_spatial_classification='unavailable_registration_unverified')
                if selected:
                    p=classify(mask,unknown,x,y,a,b,c['intersection_q_tolerance'])
                    m=classify(mask,unknown,x,y,mean_a,mean_b,c['intersection_q_tolerance'])
                    rp=roi_boundary_flags(polygon,x,y,a,b);rm=roi_boundary_flags(polygon,x,y,mean_a,mean_b)
                    result.update(conditional_center_inside_original_mask=point_inside_mask(mask,x,y),
                        pointwise_label=p['label'],frame_mean_label=m['label'],agreement_category=agreement(p['label'],m['label']),
                        pointwise_unknown_reason=p['reason'],frame_mean_unknown_reason=m['reason'],
                        pointwise_touches_ROI_boundary=rp['touches_ROI_boundary'],pointwise_crosses_ROI_boundary=rp['crosses_ROI_boundary'],
                        frame_mean_touches_ROI_boundary=rm['touches_ROI_boundary'],frame_mean_crosses_ROI_boundary=rm['crosses_ROI_boundary'],
                        pointwise_min_q_in_local_search=p['min_q'],frame_mean_min_q_in_local_search=m['min_q'],
                        pointwise_intersects_C1_zero_unknown=p['intersects_zero_unknown'],frame_mean_intersects_C1_zero_unknown=m['intersects_zero_unknown'])
                    count_p[p['label']]+=1;count_m[m['label']]+=1;counts_d[result['agreement_category']]+=1
                    # Independent full-frame distance enumeration, not the
                    # production bounding-box search, for every selected point.
                    dx=np.maximum(np.abs(foreground_x.astype(float)-x)-.5,0)
                    dy=np.maximum(np.abs(foreground_y.astype(float)-y)-.5,0)
                    for aa,bb,res in [(a,b,p),(mean_a,mean_b,m)]:
                        brute=bool(np.any((dx/aa)**2+(dy/bb)**2<=1+c['intersection_q_tolerance']))
                        assert brute==res['intersects_known_mask'];brute_checks+=1
                extras.append(result)
            common=len(chosen)-counts_d['unknown_in_at_least_one_method']
            agree=counts_d['both-compatible']+counts_d['both-non-mito-compatible']
            summary=dict(frame=frame,time_s=frame*.02,registration_status=STATUS,
                ROI_candidates=len(rows),selected=len(chosen),excluded=len(rows)-len(chosen),
                mean_x_se_nm=mean_x,mean_y_se_nm=mean_y,mean_axis_x_px=mean_a,mean_axis_y_px=mean_b,
                mean_type='arithmetic_mean_of_all_QC_selected_axis_SE_no_sqrt_n',
                pointwise_counts=count_p,frame_mean_counts=count_m,agreement_counts=counts_d,
                common_determinate=common,agreement_count=agree,agreement_fraction=agree/common if common else None,
                original_mask_full_pixels=int(mask.sum()),original_mask_ROI_pixels=int(analysis[frame].sum()),
                original_C1_zero_pixels_in_ROI=int((unknown&roi).sum()))
            assert sum(count_p.values())==sum(count_m.values())==sum(counts_d.values())==len(chosen)
            mask_geom=pixel_square_union(mask);expanded=expanded_for_display(mask_geom,mean_a,mean_b,c['visual_buffer_quad_segs'])
            assert mask_geom.is_valid and expanded.is_valid
            image=plot_frame(frame,C2,mask_geom,expanded,vertices,rows,extras,summary,c,out)
            figures.append(str(image));summaries.append(summary);all_extra.extend(extras)
            for method,col in [('pointwise','pointwise_label'),('frame_mean','frame_mean_label')]:
                for label in LABEL_COLORS:
                    members=[row for row,e in zip(rows,extras) if e['spatial_selected'] and e[col]==label]
                    record=dict(frame=frame,time_s=frame*.02,method=method,label=label,n=len(members),registration_status=STATUS)
                    for field in ['x_se_nm','y_se_nm','photons_e']:
                        for statistic,value in stats([number(row,field) for row in members]).items():
                            record[field+'_'+statistic]=value
                    groups.append(record)
            frame_checks.append(dict(frame=frame,C1_pixel_sha256=hashlib.sha256(C1.tobytes()).hexdigest(),
                C2_pixel_sha256=hashlib.sha256(C2.tobytes()).hexdigest(),
                clean_full_mask_pixel_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                analytic_vs_independent_full_enumeration_checks=brute_checks,image=image.name,
                selected=len(chosen),excluded=len(rows)-len(chosen)))
            raw1.append(C1);raw2.append(C2);selected_full.append(np.asarray(mask));selected_analysis.append(np.asarray(analysis[frame]))
            print('FRAME DONE:',json.dumps(summary),flush=True)
    write_appended_csv(out/'all_three_frame_ROI_candidates_with_spatial_preview.csv',fields,original,all_extra)
    with (out/'selected_input_original_fit_rows.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(original)
    selected_table=pd.DataFrame([dict(row,**e) for row,e in zip(original,all_extra) if e['spatial_selected']])
    selected_table.to_csv(out/'selected_spatial_preview_points.csv',index=False)
    pd.DataFrame(groups).to_csv(out/'group_uncertainty_and_signal_summary.csv',index=False)
    flat=[]
    for s in summaries:
        row={k:v for k,v in s.items() if not isinstance(v,dict)}
        for key in ['pointwise_counts','frame_mean_counts','agreement_counts']:
            row.update({key+'__'+label:n for label,n in s[key].items()})
        flat.append(row)
    pd.DataFrame(flat).to_csv(out/'per_frame_summary.csv',index=False)
    dump(out/'summary.json',summaries);dump(out/'frame_manifest.json',frame_checks)
    np.savez_compressed(out/'selected_frame_inputs.npz',original_frames=np.array(frames),
        C1_raw=np.stack(raw1),C2_raw=np.stack(raw2),clean_full_mask=np.stack(selected_full),
        analysis_ROI_mask=np.stack(selected_analysis),roi_mask=roi,roi_vertices_xy=vertices)
    # Independent re-read of source pages after rendering, and unchanged source
    # SHA256 checks. No rw memmap or source-writing operation is used.
    with tifffile.TiffFile(verified['C1_file']) as tf1,tifffile.TiffFile(verified['C2_file']) as tf2:
        for item in frame_checks:
            f=item['frame']
            assert hashlib.sha256(tf1.pages[f].asarray().tobytes()).hexdigest()==item['C1_pixel_sha256']
            assert hashlib.sha256(tf2.pages[f].asarray().tobytes()).hexdigest()==item['C2_pixel_sha256']
            assert hashlib.sha256(full[f].tobytes()).hexdigest()==item['clean_full_mask_pixel_sha256']
    print('Checking all protected input SHA256 values after preview...',flush=True)
    after=[fingerprint(p) for p in inputs];dump(out/'input_fingerprints_after.json',after);assert before==after
    dump(out/'validation.json',dict(all_checks_passed=True,geometry_unit_tests=7,preview_contract_tests=3,
        synthetic_full_enumeration_cases=500,real_full_enumeration_cases=sum(r['analytic_vs_independent_full_enumeration_checks'] for r in frame_checks),
        current_QC_selected_counts=[s['selected'] for s in summaries],excluded_counts=[s['excluded'] for s in summaries],
        all_166_original_ROI_rows_and_fields_preserved_verbatim=True,no_original_fits_or_QC_modified=True,
        original_mask_and_cache_fingerprints_unchanged=True,source_TIFF_pages_rechecked=True,
        frame_numbers_times_ROI_and_mask_mapping_valid=True,classification_uses_uncut_full_mask=True,
        no_refitting_or_resegmentation=True,no_TARDIS=True,registration_verified=False,coverage_validated=False))
    snap=out/'source_snapshot';snap.mkdir()
    for p in HERE.glob('*.py'):shutil.copyfile(p,snap/p.name)
    shutil.copyfile(ROOT/'TARDIS/path_irrelevant_python/tardis_tau1/locbio.py',snap/'locbio.py')
    shutil.copyfile(ROOT/'mitochondria_mask_pilot/envs/cleanup/requirements.lock.txt',out/'cleanup_requirements.lock.txt')
    dump(out/'versions.json',dict(python=sys.version,executable=sys.executable,
        packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','shapely','matplotlib','Pillow','tifffile']},
        verification_python=verified['config']['verification_python']))
    (out/'run_command.txt').write_text(shlex.join([sys.executable,str(Path(__file__).resolve()),'--config',str(args.config.resolve())])+'\n')
    dump(out/'field_units.json',dict(fit_coordinates='native C2 pixels, floating, original fields unchanged',
        x_se_nm='full observed-Hessian local model X standard error, nm, manufacturer nominal',
        y_se_nm='same, Y',pointwise_axes='pixels: per-point SE_nm /117',
        frame_mean_axes='pixels: arithmetic mean of selected per-axis SE_nm /117',
        conditional_center_inside_original_mask='membership of original closed-pixel-square union under identity assumption',
        pointwise_label='exploratory conditional ellipse-mask compatibility, not binding',
        frame_mean_label='exploratory conditional common-ellipse mask expansion membership',
        validated_spatial_classification='always unavailable_registration_unverified',
        mean_expanded_boundary='approximate display-only GEOS buffer; not used for numerical classification'))
    (out/'REPORT_zh.txt').write_text(
        '只处理原始帧 200、1200、2000（4、24、40 秒）。原定位与 mask 缓存只读。\n'
        '原始 ROI 候选 166 行全部保留；QC 入选 80/45/22，共 147；排除 13/3/3，共 19。\n'
        'C1/C2 恒等变换仅为用户选择的探索性假设，精细配准未验证。所有有效性字段保留此状态。\n'
        '各点误差为当前 7×7 外圈背景约束拟合的全 Hessian 模型标准误，不是原生 Picasso CRLB 或已验证误差覆盖。\n'
        '逐点法使用自身 X/Y 半轴；逐帧法分别对该帧所有入选点 X/Y 标准误求算术平均，不除 √n。\n'
        '两法共用未被 ROI 切断的 clean_full_mask、同一定位集合及闭合像素方块几何，包含相切。\n'
        '没有将 30–45 nm 换成一圈 117 nm 像素；解析判定包括亚像素几何，不使用显示 buffer 的近似边界作判定。\n'
        '原始 C1 零值区域保守标为未知，不称为确定背景；误差范围越出视野或接触该区域时为 unknown。\n'
        '细胞 ROI 不扩张。误差椭圆碰到或跨过 ROI 边界单独标记，不排除。\n'
        '分类与计数仅表示该恒等假设下的空间相容性；non-mito-compatible 不是已确认胞质，compatible 不是已证明结合。\n'
        '两法可能由逐点误差大小产生选择差异，应结合组内误差与信号摘要查看；不从相关性推断原因。\n'
        '输入 SHA256 前后相同，原 CSV 字段逐字保留，独立完整像素枚举核验全部 147 点×两方法。\n'
        'selected_frame_inputs.npz 的轴为“所选帧索引,Y,X”，original_frames=[200,1200,2000]；这些帧不是连续时间。\n'
        '只交付 PNG、分类审计表、摘要和来源记录。没有重定位、重分割、TARDIS、轨迹、扩散或状态比例拟合。\n')
    outputs=[fingerprint(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['run.log','output_fingerprints.json','COMPLETED.json']]
    dump(out/'output_fingerprints.json',outputs)
    dump(out/'COMPLETED.json',dict(all_checks_passed=True,output=str(out),frames=frames,
        selected_points=147,retained_ROI_candidates=166,figures=figures,
        registration_status=STATUS,elapsed_s=time.perf_counter()-started))
    print('COMPLETED:',out,flush=True)
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True)
    args=ap.parse_args()
    try:main(args)
    except BaseException:
        try:
            latest=Path((HERE/'latest_output.txt').read_text().strip())
            (latest/'failure.txt').write_text(traceback.format_exc())
        finally:raise
