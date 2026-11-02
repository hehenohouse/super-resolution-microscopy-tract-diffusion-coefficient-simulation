#!/usr/bin/env python3
"""Standalone native LoG detection and frozen annular-constrained fitting."""
import argparse,concurrent.futures,hashlib,json,multiprocessing,os,subprocess,sys,time
from collections import Counter
from pathlib import Path
from batch_common import (load_job,configure_environment,vendor_paths,guard_path,dump,sha,
    descriptor,verify_descriptor,stage_done,commit_stage,log,utc,atomic_replace,SAMPLE_FRAMES)

NATIVE_FIELDS=['spot_id','frame','time_s','x_px','y_px','x_um','y_um','quality_log_response','radius_setting_um']
FIT_FIELDS=['candidate_index','spot_id','frame','time_s','fit_attempted','fit_success','active_bound',
 'fit_usable_interior','finite_uncertainty','fit_x_px','fit_y_px','photons_e','bg_e_per_pixel',
 'sx_px','sy_px','x_se_nm','y_se_nm','N_se_e','bg_se_e_per_pixel','uncertainty_status',
 'fit_origin_x_px','fit_origin_y_px','unavailable_reason','background_supported','background_pixels',
 'background_variance_e2','background_constraint_se_e','calibration_status','quantitatively_calibrated',
 'estimator','suspect_overlap','fit_review_flags','fit_message','iterations','negative_log_likelihood',
 'captured_psf_fraction','signal_inside_box_e','refinement_shift_px','residual_rms_e',
 'standardized_residual_rms','corr_N_bg','fitted_center_inside_ROI','histogram_eligible',
 'qc_unflagged_model_fit','lpx_model_nm','lpy_model_nm','lpx_model_px','lpy_model_px','audit_status',
 'source_detection_uid','source_detection_cache','reused_from_pilot','pilot_candidate_index']
BOOL_FIELDS=['fit_attempted','fit_success','active_bound','fit_usable_interior','finite_uncertainty',
 'background_supported','quantitatively_calibrated','suspect_overlap','fitted_center_inside_ROI',
 'histogram_eligible','qc_unflagged_model_fit','reused_from_pilot']

def scientific_config(job):
    c=dict(job['fit']);reference=json.loads((Path(job['release'])/'fit_reference.json').read_text(encoding='utf-8'))
    keys=['trackmate_version','radius_um','threshold','median_filter','subpixel_localization',
      'native_detection_chunk_frames','box_px','annulus_inner_radius_px','annulus_outer_radius_px',
      'neighbor_core_radius_px','minimum_background_pixels','minimum_jackknife_sectors',
      'width_bounds_px','initial_sigma_multistart_px','fit_max_iterations','workers','pixel_size_nm',
      'frame_interval_s','calibration_status','noise_model','camera']
    for key in keys:
        if c[key]!=reference[key]:raise ValueError('Frozen localization rule differs: '+key)
    for key in ['sample_id','sample_name','C1_file','C2_file','crop_xyxy_px']:
        c[key]=job[key]
    c['roi_file']=job['ROI_file']
    return c

def imports():
    vendor_paths('real_annular_background')
    import numpy as np,pandas as pd,tifffile
    import run_real,run_full
    from parallel_fit import fit_batch
    return np,pd,tifffile,run_real,run_full,fit_batch

IDENTITY_FIELDS=['molecule','genotype','BR','BR_raw_label','cell','acquisition_date',
    'sample_id','sample_name','run_id','biological_replicate_id','biological_independence_confirmed']

def add_identity(table,job):
    """Attach mergeable sample identity without changing original fit/QC values.
    BR remains a raw folder label; biological independence has not been supplied.
    """
    identity=dict(molecule=str(job.get('molecule','TRAK1')),
        genotype=str(job['genotype']),BR=str(job['BR']),BR_raw_label=str(job['BR']),
        cell=str(job['cell']),acquisition_date=str(job['acquisition_date']),
        sample_id=str(job['sample_id']),sample_name=str(job['sample_name']),run_id=str(job['run_id']),
        biological_replicate_id=str(job.get('biological_replicate_id','')),
        biological_independence_confirmed=str(job.get('biological_independence_confirmed','')))
    result=table.copy()
    for key,value in identity.items():
        if key in result and len(result):
            if not result[key].fillna('').astype(str).eq(value).all():
                raise ValueError('Existing localization identity disagrees with job: '+key)
        result[key]=value
    return result

def atomic_csv(table,path):
    path=guard_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    table.to_csv(tmp,index=False,compression=dict(method='gzip',mtime=0))
    atomic_replace(tmp,path)

def read_csv(path):
    import pandas as pd
    return pd.read_csv(path,compression='gzip' if str(path).endswith('.gz') else None,float_precision='round_trip')

def read_vertices(path):
    import numpy as np
    path=Path(path)
    vertices=np.load(path,allow_pickle=False) if path.suffix.lower()=='.npy' else read_csv(path)[['x_px','y_px']].to_numpy(float)
    vertices=np.asarray(vertices,dtype=float)
    if vertices.ndim!=2 or vertices.shape[1]!=2 or len(vertices)<3 or not np.isfinite(vertices).all():
        raise ValueError('ROI vertices must be a finite N-by-2 original-coordinate array')
    return vertices

def detection_chunks(frames):
    """Only tests specify a subset; production always requests all 2500 frames."""
    return [(a,a+25) for a in range(0,2500,25) if any(a<=f<a+25 for f in frames)]

def desired_frames(job):
    frames=sorted(set(map(int,job.get('test_frames_only',range(2500)))))
    if not frames or min(frames)<0 or max(frames)>=2500:raise ValueError('Invalid source frame selection')
    if 'test_frames_only' in job and job.get('test_mode') is not True:
        raise ValueError('Subset localization is restricted to explicit golden tests')
    return frames

def checked_chunk(job,start,stop,source):
    p=Path(job['work'])/'localization_detection'/f'frames{start:04d}_{stop-1:04d}.json'
    if not p.exists():return None
    d=json.loads(p.read_text(encoding='utf-8'));csv=p.with_suffix('.csv')
    assert d['complete'] and d['scientific_fingerprint']==job['scientific_fingerprint']
    assert d['C2_source_descriptor']==source and verify_descriptor(source)
    assert d['start']==start and d['stop_exclusive']==stop and d['csv_sha256']==sha(csv)
    assert d['trackmate_version']=='7.14.0' and d['radius_um']==.28 and d['threshold']==6.5
    assert len(d['counts'])==stop-start and sum(d['counts'])==d['rows']
    return d

def ensure_java(job):
    # Java11 bytecode is compiled during packaging, never into a frozen release.
    root=Path(job['release'])/'java'
    classes=[root/'DriveMemoryPreferencesFactory.class',root/'DriveMemoryPreferencesFactory$MemoryNode.class']
    if not all(p.is_file() and p.stat().st_size>8 for p in classes):
        from batch_common import BatchInfrastructureError
        raise BatchInfrastructureError('Bundled portable Java preference classes are missing; repackage, do not compile during analysis')
    return classes

def detect(job):
    if stage_done(job,'detection'):
        first=detection_chunks(desired_frames(job))[0]
        checked_chunk(job,*first,descriptor(job['C2_file']))
        return
    started=time.perf_counter();c=scientific_config(job);np,pd,tifffile,core,full,_=imports()
    frames=desired_frames(job);chunks=detection_chunks(frames);source=descriptor(job['C2_file'])
    root=guard_path(Path(job['work'])/'localization_detection');root.mkdir(parents=True,exist_ok=True)
    ensure_java(job);reports=[]
    while True:
        remaining=[(a,b) for a,b in chunks if checked_chunk(job,a,b,source) is None]
        if not remaining:break
        request=root/'worker_request.json';result=root/'worker_result.json'
        dump(request,dict(chunks=remaining[:10],C2_source_descriptor=source,result=str(result),synthetic_smoke=bool(job.get('validation_only')) and remaining[0]==chunks[0]))
        command=[job['runtime']['detector_python'],'-B','-u',str(Path(job['release'])/'detector_worker.py'),
                 '--job',job['_job_path'],'--request',str(request)]
        with (root/'native.log').open('a') as handle:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                                     cwd=job['work'],env=dict(os.environ))
            for line in process.stdout:print(line,end='',flush=True);handle.write(line);handle.flush()
            if process.wait():raise RuntimeError('Native TrackMate worker failed')
        worker_record=json.loads(result.read_text(encoding='utf-8'))
        if worker_record.get('synthetic_smoke'):
            dump(Path(job['output'])/'state/native_synthetic_smoke.json',worker_record)
        after=[(a,b) for a,b in chunks if checked_chunk(job,a,b,source) is None]
        if len(after)>=len(remaining):raise RuntimeError('Native worker made no committed progress')
        log(job,'detection',completed_frames=(len(chunks)-len(after))*25,total_requested_frames=len(chunks)*25,
            elapsed_s=time.perf_counter()-started)
    tables=[];hashes=[]
    vertices=read_vertices(job['roi_vertices_path'])
    for a,b in chunks:
        info=checked_chunk(job,a,b,source);reports.append(info)
        tag=f'frames{a:04d}_{b-1:04d}';p=root/(tag+'.csv');table=read_csv(p)
        assert len(table)==info['rows']
        table['source_detection_uid']=[job['sample_id']+':'+tag+':'+str(int(v)) for v in table.spot_id]
        table['candidate_uid']=table['source_detection_uid'];table['source_detection_cache']=str(p)
        tables.append(table);hashes.extend(info['sample_page_hashes'])
    all_rows=pd.concat(tables,ignore_index=True).sort_values(['frame','x_px','y_px'],kind='stable').reset_index(drop=True)
    all_rows.insert(0,'candidate_index',np.arange(len(all_rows),dtype=np.int64))
    finite=np.isfinite(all_rows[['x_px','y_px']].to_numpy(float)).all(axis=1)
    flags=np.full(len(all_rows),-1,dtype=np.int8)
    flags[finite]=core.roi_lib.points_in_polygon(all_rows.loc[finite,['x_px','y_px']].to_numpy(),vertices).astype(np.int8)
    all_rows['roi_inside']=flags
    assert all_rows.candidate_index.is_unique and all_rows.source_detection_uid.is_unique
    all_rows=add_identity(all_rows,job)
    canonical=Path(job['output'])/'data/all_detections.csv.gz';atomic_csv(all_rows,canonical)
    stats=Path(job['output'])/'data/detection_frame_counts.csv'
    fov=all_rows.groupby('frame').size();inside=all_rows[all_rows.roi_inside==1].groupby('frame').size()
    count=pd.DataFrame([dict(frame=f,time_s=f*.02,FOV_candidates=int(fov.get(f,0)),
       ROI_candidates=int(inside.get(f,0))) for f in range(2500)])
    stats.parent.mkdir(parents=True,exist_ok=True);count.to_csv(stats,index=False)
    sample_report=Path(job['output'])/'state/localization_sample_pixels.json'
    dump(sample_report,dict(source=source,hash_method='SHA256 of raw uint16 C-order image bytes',
        sampled_frames=sorted(set(x['frame'] for x in hashes)),page_hashes=hashes))
    assert verify_descriptor(source),'Source changed during detection'
    commit_stage(job,'detection',[canonical,stats,sample_report],FOV_candidates=len(all_rows),
        ROI_candidates=int((all_rows.roi_inside==1).sum()),requested_frames=frames,elapsed_s=time.perf_counter()-started)

def empty_fits(pd):
    table=pd.DataFrame(columns=FIT_FIELDS)
    for key in BOOL_FIELDS:table[key]=table[key].astype(bool)
    return table

def fit(job):
    if stage_done(job,'localization'):
        first=detection_chunks(desired_frames(job))[0]
        checked_chunk(job,*first,descriptor(job['C2_file']))
        return
    if not stage_done(job,'detection'):raise RuntimeError('Detection stage must be committed before fitting')
    started=time.perf_counter();c=scientific_config(job);np,pd,tifffile,core,full,fit_batch=imports()
    frames=desired_frames(job);first=detection_chunks(frames)[0]
    checked_chunk(job,*first,descriptor(job['C2_file']))
    all_rows=read_csv(Path(job['output'])/'data/all_detections.csv.gz')
    vertices=read_vertices(job['roi_vertices_path']);roi=np.load(job['roi_mask_path'])
    assert roi.shape==(684,428) and roi.dtype==bool
    source=descriptor(job['C2_file']);detect_sha=sha(Path(job['output'])/'data/all_detections.csv.gz')
    root=guard_path(Path(job['work'])/'localization_fits');root.mkdir(parents=True,exist_ok=True)
    frame_groups={int(f):t for f,t in all_rows.groupby('frame',sort=False)}
    fit_tables=[];all_counts=[];sample_hashes={}
    with tifffile.TiffFile(job['C2_file']) as tf,concurrent.futures.ProcessPoolExecutor(
             max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
        for start,stop in detection_chunks(frames):
            chunk_frames=[f for f in frames if start<=f<stop];tag=f'frames{start:04d}_{stop-1:04d}'
            marker=root/(tag+'.json');target=root/(tag+'.csv.gz');counts_path=root/(tag+'_counts.csv.gz')
            identity=dict(scientific_fingerprint=job['scientific_fingerprint'],source_descriptor=source,
                detection_sha256=detect_sha,frames=chunk_frames)
            if marker.exists():
                saved=json.loads(marker.read_text(encoding='utf-8'));assert saved['identity']==identity and saved['complete']
                assert sha(target)==saved['fit_sha256'] and sha(counts_path)==saved['counts_sha256']
                tab=read_csv(target);counts=read_csv(counts_path)
                assert len(tab)==saved['rows'];sample_hashes.update(saved.get('sample_page_hashes',{}))
            else:
                parts=[];records=[]
                for f in chunk_frames:
                    table=frame_groups.get(f,pd.DataFrame(columns=all_rows.columns)).copy()
                    targets=table[table.roi_inside==1].copy();raw=tf.pages[f].asarray()
                    if f in job['sample_frames']:
                        sample_hashes[str(f)]=hashlib.sha256(raw.tobytes(order='C')).hexdigest()
                    if len(targets):
                        bg,masks,movie=core.background_geometry(targets,table,raw,roi,c,c['camera'])
                        items=[(int(r.candidate_index),int(r.spot_id),f,float(r.x_px),float(r.y_px)) for r in targets.itertuples()]
                        step=max(1,int(np.ceil(len(items)/4)))
                        tasks=[pool.submit(fit_batch,(items[a:a+step],raw,movie,
                            bg.iloc[a:a+step].reset_index(drop=True),masks[a:a+step],c))
                            for a in range(0,len(items),step)]
                        values=[r for future in tasks for r in future.result()]
                        fits=full.decorate(pd.DataFrame(values),vertices,c)
                        # Preserve fitter fields and original detector coordinates without overwriting.
                        extras=[k for k in targets.columns if k not in fits.columns]
                        fits=fits.merge(targets[['candidate_index']+extras],on='candidate_index',how='left',validate='one_to_one',sort=False)
                        for k in bg.columns:
                            if k=='candidate_index':continue
                            key=k if k not in fits.columns else 'annular_'+k
                            fits[key]=bg[k].to_numpy()
                        fits['reused_from_pilot']=False;fits['pilot_candidate_index']=np.nan
                        assert fits.source_detection_uid.tolist()==targets.source_detection_uid.tolist()
                    else:fits=empty_fits(pd)
                    parts.append(fits);reason=Counter(fits.audit_status)
                    records.append(dict(frame=f,time_s=f*.02,FOV_candidates=len(table),ROI_candidates=len(fits),
                        selected=int(fits.qc_unflagged_model_fit.sum()),
                        QC_excluded=int((~fits.qc_unflagged_model_fit).sum()),
                        **{'QC_reason__'+k:int(v) for k,v in reason.items() if k!='unflagged_model_fit'}))
                tab=pd.concat(parts,ignore_index=True);counts=pd.DataFrame(records).fillna(0)
                atomic_csv(tab,target);atomic_csv(counts,counts_path)
                dump(marker,dict(complete=True,identity=identity,rows=len(tab),fit_sha256=sha(target),
                    counts_sha256=sha(counts_path),sample_page_hashes=sample_hashes,completed_utc=utc()))
            fit_tables.append(tab);all_counts.append(counts)
            completed=sum(len(v) for v in all_counts)
            if completed%250==0 or completed==len(frames):
                log(job,'localization',completed_frames=completed,total_frames=len(frames),
                    ROI_candidates=sum(len(v) for v in fit_tables),
                    selected=sum(int(v.qc_unflagged_model_fit.sum()) for v in fit_tables),
                    elapsed_s=time.perf_counter()-started)
    result=pd.concat(fit_tables,ignore_index=True)
    counts=pd.concat(all_counts,ignore_index=True).sort_values('frame')
    assert result.candidate_index.is_unique and result.source_detection_uid.is_unique
    expected=all_rows[(all_rows.roi_inside==1)&all_rows.frame.isin(frames)]
    np.testing.assert_array_equal(result.candidate_index,expected.candidate_index)
    np.testing.assert_array_equal(result.source_detection_uid,expected.source_detection_uid)
    assert len(result)==int(counts.ROI_candidates.sum())
    assert int(result.qc_unflagged_model_fit.sum())==int(counts.selected.sum())
    assert verify_descriptor(source),'Source changed during localization'
    for row in json.loads((Path(job['output'])/'state/localization_sample_pixels.json').read_text(encoding='utf-8'))['page_hashes']:
        if row['frame'] in frames:assert sample_hashes[str(row['frame'])]==row['pixel_sha256']
    result=add_identity(result,job)
    canonical=Path(job['output'])/'data/roi_fits.csv.gz';atomic_csv(result,canonical)
    count_path=Path(job['output'])/'data/localization_frame_counts.csv';counts.to_csv(count_path,index=False)
    summary=Path(job['output'])/'data/localization_summary.json'
    dump(summary,dict(sample_id=job['sample_id'],frames=len(frames),FOV_candidates=int(counts.FOV_candidates.sum()),
        ROI_candidates=len(result),selected=int(result.qc_unflagged_model_fit.sum()),
        QC_excluded=int((~result.qc_unflagged_model_fit).sum()),QC_reasons=dict(Counter(result.audit_status)),
        calibration_status='manufacturer_nominal',model_SE_coverage_validated=False,
        original_numerical_functions=['run_real.background_geometry','run_real.fit_one','run_full.decorate'],
        annular_pixel_NPZ_saved=False))
    commit_stage(job,'localization',[canonical,count_path,summary],ROI_candidates=len(result),
        selected=int(result.qc_unflagged_model_fit.sum()),QC_excluded=int((~result.qc_unflagged_model_fit).sum()),
        elapsed_s=time.perf_counter()-started)

def bind_validation_reference(job):
    """Use saved all-FOV candidate identities for rerun fits; actual LoG stays separate."""
    np,pd,_,_,_,_=imports()
    reference=Path(job['validation_reference_dir'])/'expected_detections.csv.gz'
    if not reference.is_file():raise RuntimeError('Missing validation all-FOV detection reference')
    root=Path(job['output']);data=root/'data';binding=root/'state/validation_candidate_binding.json'
    actual=data/'actual_native_detections.csv.gz'
    canonical=data/'all_detections.csv.gz'
    expected_sha=sha(reference)
    if binding.exists():
        saved=json.loads(binding.read_text(encoding='utf-8'))
        if saved['reference_SHA256']!=expected_sha or sha(canonical)!=saved['canonical_SHA256'] or sha(actual)!=saved['actual_native_SHA256']:
            raise RuntimeError('Validation candidate binding changed')
        return
    # Read/write through the canonical portable CSV routine, never edit source fixtures.
    actual_rows=read_csv(canonical);atomic_csv(actual_rows,actual)
    baseline=read_csv(reference)
    required={'candidate_index','spot_id','frame','time_s','x_px','y_px','source_detection_uid'}
    if not required<=set(baseline.columns):raise RuntimeError('Validation reference omits candidate identity/coordinates')
    baseline=baseline[baseline.frame.isin(SAMPLE_FRAMES)].copy().sort_values('candidate_index',kind='stable')
    if not baseline.candidate_index.is_unique or not baseline.source_detection_uid.is_unique:
        raise RuntimeError('Validation reference identities are not unique')
    vertices=read_vertices(job['roi_vertices_path'])
    _,_,_,core,_,_=imports()
    baseline['roi_inside']=core.roi_lib.points_in_polygon(baseline[['x_px','y_px']].to_numpy(float),vertices).astype('int8')
    for field in IDENTITY_FIELDS:
        if field in baseline:baseline['validation_reference_'+field]=baseline.pop(field)
    baseline=add_identity(baseline,job)
    atomic_csv(baseline,canonical)
    counts=data/'detection_frame_counts.csv'
    if counts.exists():
        import shutil
        shutil.copyfile(counts,data/'actual_native_detection_frame_counts.csv')
    fov=baseline.groupby('frame').size();inside=baseline[baseline.roi_inside==1].groupby('frame').size()
    pd.DataFrame([dict(frame=f,time_s=f*.02,FOV_candidates=int(fov.get(f,0)),ROI_candidates=int(inside.get(f,0)))
                  for f in range(2500)]).to_csv(counts,index=False)
    report=dict(all_passed=True,reference=str(reference),reference_SHA256=expected_sha,
        actual_native_SHA256=sha(actual),canonical_SHA256=sha(canonical),
        saved_reference_candidates_used_for_actual_fitting=True,cached_fit_values_used=False,
        native_SpotIDs_not_assumed_reproducible=True,frames=SAMPLE_FRAMES)
    dump(binding,report)
    commit_stage(job,'detection',[canonical,actual,counts,root/'state/localization_sample_pixels.json',binding],
       validation_only=True,FOV_candidates=len(baseline),actual_native_candidate_count=len(actual_rows))

def validation_math_and_empty(job):
    np,pd,_,core,_,_=imports()
    math=core.math_checks()
    smoke=json.loads((Path(job['output'])/'state/native_synthetic_smoke.json').read_text(encoding='utf-8'))
    assert smoke['synthetic_smoke']['counts']==[2,0,2]
    # Known integrated Gaussian, rounded once into uint16 ADU. Backend-only
    # deterministic coordinate/finite-covariance control; not coverage calibration.
    c=scientific_config(job);yy,xx=np.mgrid[:64,:80];truth=(40.25,32.4)
    xy=np.column_stack([xx.ravel(),yy.ravel()])
    psf=core.arbitrary_psf(xy,*truth,1.4,1.4)[0].reshape(64,80)
    raw=np.rint(406.+800.*psf/.46).astype(np.uint16)
    targets=pd.DataFrame([dict(candidate_index=0,spot_id=0,frame=0,x_px=truth[0],y_px=truth[1])])
    bg,masks,movie=core.background_geometry(targets,targets,raw,np.ones(raw.shape,bool),c,c['camera'])
    value=core.fit_one((0,0,0,*truth),raw,movie,bg.iloc[0],masks[0],c)
    assert value['fit_success'] and value['fit_usable_interior'] and value['finite_uncertainty']
    np.testing.assert_allclose([value['fit_x_px'],value['fit_y_px']],truth,atol=.15,rtol=0)
    assert np.isfinite([value['x_se_nm'],value['y_se_nm']]).all()
    report=dict(all_passed=True,actual_native_blank_checked=True,actual_native_known_spots_checked=True,
        actual_synthetic_fit_checked=True,math_checks=math,native_smoke=smoke['synthetic_smoke'],
        synthetic_fit=dict(truth_xy_px=truth,expected_photoelectrons=800.,result=value,
            coordinate_absolute_tolerance_px=.15,role='Known-position backend check; not precision coverage validation'),
        actual_real_fit_frames=SAMPLE_FRAMES,model_and_QC_unchanged=True)
    dump(Path(job['output'])/'state/validation_math_and_empty.json',report)

def run(job,stage='all'):
    configure_environment(job['runtime_dir']);scientific_config(job)
    if stage=='validate-windows':
        if not (job.get('validation_only') is True and job.get('test_mode') is True and job.get('test_frames_only')==SAMPLE_FRAMES):
            raise RuntimeError('Windows qualification requires explicit fixed10 validation mode')
        detect(job);bind_validation_reference(job);fit(job);validation_math_and_empty(job);return
    if stage in ['detect','all']:detect(job)
    if stage in ['fit','all']:fit(job)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',required=True,type=Path)
    ap.add_argument('--stage',choices=['detect','fit','all','validate-windows'],default='all');args=ap.parse_args()
    job=load_job(args.job);run(job,args.stage)

if __name__=='__main__':main()
