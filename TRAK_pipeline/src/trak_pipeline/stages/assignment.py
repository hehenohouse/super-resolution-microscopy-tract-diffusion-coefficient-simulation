"""UID-keyed k=1 uncertainty-ellipse assignment, independent of mask producer.

Input localization master remains read-only. Only labels and compact counts are
written. Unknown C1 pixels, missing support blocks, and closed-FOV boundaries
have precedence over positive mask contact.
"""
from __future__ import annotations
from collections import Counter, defaultdict, OrderedDict
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

from .mask import _input, _file, _load_contract, _validate_contract, _acquisition, _json, _read, _close, _result, _temporary_path

LABELS=('mito-compatible','non-mito-compatible','unknown')
FIELDS=['source_detection_uid','frame','spatial_selected','pointwise_label','pointwise_unknown_reason',
        'mask_block_index','pointwise_axis_x_px','pointwise_axis_y_px','pointwise_intersects_known_mask',
        'pointwise_intersects_C1_zero_unknown','pointwise_min_q']
REQUIRED=('source_detection_uid','frame','fit_x_px','fit_y_px','x_se_nm','y_se_nm',
          'qc_unflagged_model_fit','fitted_center_inside_ROI','uncertainty_status')


def _flag(value):return str(value).strip().lower() in ('1','true')


def _geometry(package_root):
    path=Path(package_root)/'src/trak_pipeline/compat/vendor/spatial_mask_preview/geometry.py'
    spec=importlib.util.spec_from_file_location('_trak_frozen_ellipse_geometry',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _roi_polygon(package_root):
    """The same ray-crossing polygon test the fit stage uses for fitted_center_inside_ROI."""
    path=Path(package_root)/'src/trak_pipeline/compat/vendor/TARDIS/path_irrelevant_python/tardis_tau1/locbio.py'
    name='_trak_frozen_roi_polygon'
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module  # locbio defines dataclasses, which resolve their module via sys.modules
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None);raise
    return module


def _open_table(path):
    path=Path(path)
    return gzip.open(path,'rt',encoding='utf-8',newline='') if path.suffix=='.gz' else path.open('r',encoding='utf-8',newline='')


def independent_label(mask,zero,x,y,a,b,tolerance):
    """Full-pixel enumeration for approved representative frames only."""
    import numpy as np
    if not np.isfinite([x,y,a,b]).all() or min(a,b)<=0:return 'unknown'
    h,w=mask.shape
    if x-a<-.5-tolerance or x+a>w-.5+tolerance or y-b<-.5-tolerance or y+b>h-.5+tolerance:return 'unknown'
    hits=[]
    for area in (mask,zero):
        py,px=np.nonzero(area)
        dx=np.maximum(np.abs(px.astype(float)-x)-.5,0)
        dy=np.maximum(np.abs(py.astype(float)-y)-.5,0)
        hits.append(bool(np.any((dx/a)**2+(dy/b)**2<=1+tolerance)))
    return 'unknown' if hits[1] else (LABELS[0] if hits[0] else LABELS[1])


def classify_one(row, mask, raw, contract, scale, tolerance, geometry, frame_reason=''):
    import numpy as np
    uid=str(row['source_detection_uid']);frame=int(row['frame'])
    keep=_flag(row['qc_unflagged_model_fit']) and _flag(row['fitted_center_inside_ROI'])
    block=contract['frame_to_block'].get(str(frame),-1)
    result=dict(source_detection_uid=uid,frame=frame,spatial_selected=keep,
        pointwise_label='excluded_qc',pointwise_unknown_reason='',mask_block_index=block,
        pointwise_axis_x_px='',pointwise_axis_y_px='',pointwise_intersects_known_mask='',
        pointwise_intersects_C1_zero_unknown='',pointwise_min_q='')
    if not keep:return result
    if 'full_observed_Hessian' not in row.get('uncertainty_status',''):
        raise ValueError('Missing or unrecognized standard-error semantics for UID '+uid)
    try:x,y,a,b=[float(row[k]) for k in ('fit_x_px','fit_y_px','x_se_nm','y_se_nm')]
    except (ValueError,TypeError) as error:raise ValueError('Missing localization/error values for UID '+uid) from error
    a/=scale;b/=scale
    result['pointwise_axis_x_px']=a if math.isfinite(a) else ''
    result['pointwise_axis_y_px']=b if math.isfinite(b) else ''
    if frame_reason:
        result.update(pointwise_label='unknown',pointwise_unknown_reason=frame_reason);return result
    if mask is None:
        result.update(pointwise_label='unknown',pointwise_unknown_reason='missing_mask_block');return result
    if raw is None:
        result.update(pointwise_label='unknown',pointwise_unknown_reason='missing_original_C1_frame');return result
    if raw.shape!=mask.shape:raise ValueError('Original C1 and full mask have different coordinate grids')
    answer=geometry.classify(mask,raw==0,x,y,a,b,tolerance)
    result.update(pointwise_label=answer['label'],pointwise_unknown_reason=answer['reason'],
        pointwise_intersects_known_mask=answer['intersects_known_mask'],
        pointwise_intersects_C1_zero_unknown=answer['intersects_zero_unknown'],
        pointwise_min_q='' if answer['min_q'] is None else answer['min_q'])
    return result


def _write_csv(path,rows,fields=None):
    path=Path(path);tmp=_temporary_path(path)
    try:
        with tmp.open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields or list(rows[0]));writer.writeheader();writer.writerows(rows)
        os.replace(tmp,path)
    finally:tmp.unlink(missing_ok=True)


def _input_rows(path,expected_frames,dt):
    """Keep just classification inputs in memory; no duplicated fit-master output."""
    frame_rows=defaultdict(list);seen=set();source_fields=None
    with _open_table(path) as stream:
        reader=csv.DictReader(stream);source_fields=reader.fieldnames or []
        absent=set(REQUIRED)-set(source_fields)
        if absent:raise ValueError('Localization master lacks fields: '+', '.join(sorted(absent)))
        for original in reader:
            uid=original['source_detection_uid']
            if not uid or uid in seen:raise ValueError('Empty/duplicate localization UID: '+uid)
            seen.add(uid)
            if original.get('candidate_uid') not in (None,'',uid):raise ValueError('Candidate/detection UID correspondence differs: '+uid)
            f=int(original['frame'])
            if f not in expected_frames or float(original['frame'])!=f:raise ValueError('Localization original frame outside declared acquisition')
            if original.get('time_s') not in (None,'') and not math.isclose(float(original['time_s']),f*dt,abs_tol=1e-9):raise ValueError('Localization frame/time mismatch')
            frame_rows[f].append({key:original[key] for key in REQUIRED})
    return frame_rows,len(seen),source_fields


def execute_assignment(ctx):
    import numpy as np
    import tifffile
    from trak_pipeline.contracts import unknown_reason_for_frame
    p=ctx.parameters
    if float(p.get('k',1))!=1:raise ValueError('Current explicit assignment method supports k=1 only')
    method=p.get('method',p.get('method_id','error_ellipse_k1'))
    if method not in ('error_ellipse_k1','ellipse_k1','ellipse_pixel_contact_k1','k1_error_ellipse'):
        raise ValueError('Assignment method must explicitly be the original 1SE ellipse')
    if p.get('closed_pixel_squares',True) is not True:raise ValueError('Current method requires closed pixel squares')
    if p.get('diagonal',True) is not True or p.get('covariance','diagonal_axis_aligned')!='diagonal_axis_aligned':raise ValueError('Current method requires diagonal XY covariance approximation')
    if p.get('unknown_priority',True) is not True:raise ValueError('Unknown regions must retain precedence over mask contact')
    tolerance=float(p['tolerance'])
    if tolerance<0 or not math.isfinite(tolerance):raise ValueError('Invalid ellipse tolerance')
    shape,frames,scale,dt=_acquisition(ctx)
    cleanup=_input(ctx,'mask_cleanup','cleanup');contract=_load_contract(cleanup)
    contract=_validate_contract(contract,scale=scale,dt=dt)
    if contract['shape'][1:]!=list(shape) or list(contract['crop_offset_xy_px'])!=[0,0]:
        raise ValueError('Full-image identity assignment requires zero crop offset and the original image grid')
    fits=_input(ctx,'fit','localization','localizations');fit_path=_file(fits,'table','master','fits')
    frame_rows,nrows,source_fields=_input_rows(fit_path,set(frames),dt)
    c1=_input(ctx,'C1','c1');c1_path=_file(c1,'image','tiff')
    declared=c1.metadata.get('original_frame_ids',frames)
    if len(set(declared))!=len(declared):raise ValueError('Duplicate original C1 frame IDs')
    raw_page_for={int(f):i for i,f in enumerate(declared)}
    mask=np.load(_file(cleanup,'full_mask','clean_full_mask'),mmap_mode='r',allow_pickle=False)
    if mask.dtype!=bool or list(mask.shape)!=contract['shape']:raise ValueError('Full-mask array differs from spatial/time contract')
    roi_artifact=_input(ctx,'roi','ROI');roi=np.load(_file(roi_artifact,'mask','roi_mask'),allow_pickle=False)
    if roi.dtype!=bool or roi.shape!=shape:raise ValueError('ROI grid differs from original image grid')
    geom=_geometry(ctx.package_root)
    # Spatial selection (fit QC) uses the continuous ROI polygon. A refined center inside the
    # polygon can lie in a boundary pixel whose center is outside it (raster value 0), so the
    # selection check uses the same polygon test. The raster is used only if no polygon exists.
    vertices_path=roi_artifact.files.get('vertices')
    roi_vertices=np.load(vertices_path,allow_pickle=False) if vertices_path else None
    polygon=_roi_polygon(ctx.package_root) if roi_vertices is not None else None
    def selected_center_inside_roi(x,y):
        if roi_vertices is None:return geom.point_inside_mask(roi,x,y,tolerance)
        return bool(polygon.points_in_polygon(np.array([[x,y]],dtype=float),roi_vertices)[0])
    out=ctx.output_dir;out.mkdir(parents=True,exist_ok=True);parts=out/'parts';parts.mkdir(exist_ok=True)
    checked_frames=list(ctx.config.get('sample_frames',ctx.config.get('checking',{}).get('frames',[])))
    if not checked_frames:
        from trak_pipeline.contracts import TEN_FRAMES
        checked_frames=[f for f in TEN_FRAMES if f in frames]
    checks=set(checked_frames);independent_checked=0;all_counts=[];all_reasons=Counter();labels_parts=[]
    checkpoint_frames=int(ctx.config.get('checkpoint_frames',250))
    if checkpoint_frames<=0:raise ValueError('checkpoint_frames must be positive')
    binding=dict(fit_id=fits.id,mask_id=cleanup.id,C1_id=c1.id,ROI_id=roi_artifact.id,parameters=p,sample_frames=checked_frames)
    try:
        with tifffile.TiffFile(c1_path) as movie:
            if len(movie.pages)!=len(declared):raise ValueError('C1 pages differ from explicit original frame mapping')
            for start in range(0,len(frames),checkpoint_frames):
                ids=frames[start:start+checkpoint_frames];part=parts/('labels_%06d.csv.gz'%start);record_path=parts/('counts_%06d.json'%start)
                if part.exists() and record_path.exists():
                    saved=_read(record_path)
                    if saved.get('binding')!=binding or saved.get('frame_ids')!=ids:raise ValueError('Assignment checkpoint inputs changed')
                    all_counts.extend(saved['counts']);all_reasons.update(saved['unknown_reasons']);independent_checked+=saved['independent_checked']
                    labels_parts.append(part);continue
                counts=[];reasons=Counter();checked=0
                temporary=_temporary_path(part)
                try:
                    with gzip.open(temporary,'wt',encoding='utf-8',newline='') as stream:
                        writer=csv.DictWriter(stream,fieldnames=FIELDS);writer.writeheader()
                        for f in ids:
                            frame_reason=unknown_reason_for_frame(contract,f)
                            b=contract['frame_to_block'].get(str(f),-1)
                            full=mask[b] if not frame_reason and 0<=b<len(mask) else None
                            raw=movie.pages[raw_page_for[f]].asarray() if f in raw_page_for else None
                            if raw is not None and (raw.shape!=shape or raw.dtype!=np.uint16):raise ValueError('Original C1 grid/type mismatch')
                            tally=Counter();rows=frame_rows.get(f,[])
                            for row in rows:
                                label=classify_one(row,full,raw,contract,scale,tolerance,geom,frame_reason)
                                writer.writerow(label);tally[label['pointwise_label']]+=1
                                if label['pointwise_label']=='unknown':reasons[label['pointwise_unknown_reason']]+=1
                                if f in checks and label['spatial_selected'] and not frame_reason and raw is not None:
                                    x,y=float(row['fit_x_px']),float(row['fit_y_px'])
                                    a,baxis=float(row['x_se_nm'])/scale,float(row['y_se_nm'])/scale
                                    if not selected_center_inside_roi(x,y):raise ValueError('Selected refined center lies outside supplied ROI: '+row['source_detection_uid'])
                                    expected=independent_label(full,raw==0,x,y,a,baxis,tolerance)
                                    if expected!=label['pointwise_label']:raise AssertionError('Independent pixel enumeration disagrees: '+row['source_detection_uid'])
                                    checked+=1
                            selected=len(rows)-tally['excluded_qc']
                            count=dict(frame=f,time_s=f*dt,block_index=contract['frame_to_block'].get(str(f),-1),
                                ROI_candidates=len(rows),selected=selected,QC_excluded=tally['excluded_qc'],
                                On=tally[LABELS[0]],Not_on=tally[LABELS[1]],unknown=tally['unknown'])
                            if count['On']+count['Not_on']+count['unknown']!=selected:raise AssertionError('Assignment count conservation failed')
                            counts.append(count)
                    os.replace(temporary,part)
                finally:temporary.unlink(missing_ok=True)
                _json(record_path,dict(binding=binding,frame_ids=ids,counts=counts,unknown_reasons=dict(reasons),independent_checked=checked))
                all_counts.extend(counts);all_reasons.update(reasons);independent_checked+=checked;labels_parts.append(part)
                _json(out/'checkpoint.json',dict(completed_frames=start+len(ids),expected_frames=len(frames)))
                ctx.progress('Spatial assignment checkpoint',completed_frames=start+len(ids),expected_frames=len(frames))
    finally:_close(mask)
    labels=out/'labels.csv.gz';temp=_temporary_path(labels)
    try:
        with gzip.open(temp,'wt',encoding='utf-8',newline='') as dest:
            writer=csv.DictWriter(dest,fieldnames=FIELDS);writer.writeheader()
            for part in labels_parts:
                with gzip.open(part,'rt',encoding='utf-8',newline='') as stream:writer.writerows(csv.DictReader(stream))
        os.replace(temp,labels)
    finally:temp.unlink(missing_ok=True)
    frame_counts=out/'per_frame_counts.csv';_write_csv(frame_counts,all_counts)
    totals={key:sum(r[key] for r in all_counts) for key in ('On','Not_on','unknown','selected','QC_excluded','ROI_candidates')}
    if totals['ROI_candidates']!=nrows or totals['selected']+totals['QC_excluded']!=nrows:raise AssertionError('Assignment master-row conservation failed')
    blocks=[]
    for b,group in enumerate(contract['block_frame_ids']):
        members=[r for r in all_counts if r['block_index']==b]
        blocks.append(dict(block_index=b,**{key:sum(r[key] for r in members) for key in totals},valid=contract['block_valid'][b],unknown_reason=contract['block_reasons'][b]))
    block_counts=out/'per_block_counts.csv';_write_csv(block_counts,blocks)
    summary=out/'classification_summary.json'
    info=dict(sample_id=ctx.sample_id,**totals,counts=totals,unknown_reasons=dict(all_reasons),
        registration_status='identity/unverified',calibration_status='nominal_model_not_independently_calibrated',
        assignment='original_1SE_diagonal_ellipse_contact_with_closed_pixel_squares',pixel_size_nm=scale,frame_interval_s=dt,
        source_fit_artifact_id=fits.id,full_mask_artifact_id=cleanup.id,
        original_fit_master_read_only=True,UID_keyed_labels=True,independent_check_frames=checked_frames,
        independent_checked_selected=independent_checked,all_timepoints_independently_enumerated=False)
    _json(summary,info)
    return _result(dict(labels=labels,summary=summary,per_frame=frame_counts,per_block=block_counts),
        dict(counts=totals,source_master=str(fit_path),UID_field='source_detection_uid',uid_field='source_detection_uid',label_field='pointwise_label',selected_field='spatial_selected',contract=contract,independent_check_frames=checked_frames))
