#!/usr/bin/env python3
"""Isolated UID-bound TARDIS rerun; read-only master, saved replacement labels.

Uses a small frozen copy of the modular stages and exact frozen numerical
kernels. This entry never detects, fits, segments, links, or changes defaults.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from datetime import datetime, timezone

def utc():
    return datetime.now(timezone.utc).isoformat()

def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k,v in x.items()}
    if isinstance(x, (tuple,list)): return [safe(v) for v in x]
    if isinstance(x, Path): return str(x)
    if hasattr(x,'tolist'): return safe(x.tolist())
    if isinstance(x,float) and not math.isfinite(x): return None
    return x

def dump(p,x):
    import uuid
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temp.write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n',encoding='utf-8')
        os.replace(temp,p)
    finally: temp.unlink(missing_ok=True)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def fullsha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def descriptor(p):
    p=Path(p);s=p.stat()
    h=hashlib.sha256()
    with p.open('rb') as f:
        if s.st_size<=2*1024*1024:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            kind='complete_small_file'
        else:
            h.update(f.read(65536));f.seek(max(0,s.st_size-65536));h.update(f.read(65536))
            kind='first_last_64KiB'
    return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sampled_sha256=h.hexdigest(),fingerprint_kind=kind)

def identity(version,parameters,upstream):
    # Source hashes are audit-only, never the scientific cache identity.
    v=dict(method_version=version,parameters=parameters,upstream=upstream)
    return version+':'+hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def freeze(package,run):
    relative=[
      'src/trak_pipeline/__init__.py','src/trak_pipeline/types.py',
      'src/trak_pipeline/kernels.py','src/trak_pipeline/io.py',
      'src/trak_pipeline/stages/__init__.py','src/trak_pipeline/stages/distances.py',
      'src/trak_pipeline/stages/tardis.py','src/trak_pipeline/compat/analysis_stage.py',
      'src/trak_pipeline/compat/tardis_reference.json',
      'src/trak_pipeline/compat/vendor/picasso_full_sensitivity/full_core.py',
      'src/trak_pipeline/compat/vendor/picasso_roi_sensitivity/roi_core.py',
      'src/trak_pipeline/compat/vendor/mito_tardis/mito_core.py',
      'src/trak_pipeline/compat/vendor/mito_tardis/summarize_intra.py',
      'src/trak_pipeline/compat/vendor/mito_tardis/postmin32_extras.py']
    target=run/'source/frozen_tardis'
    entries=[]
    for name in relative:
        src=package/name;dest=target/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():
            if fullsha(dest)!=fullsha(src):
                raise ValueError('Frozen numerical source differs from installed source: '+name)
        else:shutil.copyfile(src,dest)
        entries.append(dict(relative=name,source=str(src),frozen=str(dest),SHA256=fullsha(dest)))
    dump(run/'source/tardis_source_manifest.json',dict(created_utc=utc(),audit_only=True,files=entries))
    return target

def export_comparison(run,baseline,new,c,np,pd):
    """Save scientific signed/area/positive-peak derivatives for read-only plotting."""
    scopes=['full']+[f'frames{s:04d}_{s+249:04d}' for s in range(0,2500,250)]
    # Scope names are defined by the exact frozen kernel, not guessed from filenames.
    from trak_pipeline.stages.distances import kernel
    scopes=[scope for scope,_,_ in kernel().scopes(c)]
    outputs=[];metrics=[]
    for method,root in [('A_default',baseline/'data/tardis'),('B_strong_union',new)]:
        summary=pd.read_csv(root/'per_scope_summary.csv',float_precision='round_trip')
        for group in ['On','Not-on']:
            for scope in scopes:
                frame=pd.read_csv(root/'distribution'/group/(scope+'.csv'),float_precision='round_trip')
                match=summary[(summary.group==group)&(summary.scope==scope)]
                if len(match)!=1:raise ValueError('Missing/duplicate Intra scalar status '+method+' '+group+' '+scope)
                row=match.iloc[0]
                signed=frame.intra_density_um_inv.to_numpy(float)
                area=frame.shape_density_um_inv.to_numpy(float)
                width=(frame.distance_right_um-frame.distance_left_um).to_numpy(float)
                signed_area=float(row.signed_area)
                finite=bool(np.isfinite(signed).all())
                positive_peak=float(np.max(signed)) if finite else float('nan')
                valid_intra=str(row['valid']).lower() in ('true','1')
                peak_valid=valid_intra and finite and positive_peak>0
                peak=signed/positive_peak if peak_valid else np.full(len(signed),np.nan)
                peak_reason='available_positive_peak' if peak_valid else ('unavailable_original_intra' if not valid_intra else 'unavailable_nonfinite_or_nonpositive_peak')
                if finite and not np.isclose(float(np.sum(signed*width)),signed_area,rtol=1e-11,atol=1e-12):
                    raise ValueError('Saved signed density/area disagree '+method+' '+group+' '+scope)
                d=pd.DataFrame(dict(method=method,group=group,scope=scope,
                    distance_left_um=frame.distance_left_um,distance_right_um=frame.distance_right_um,
                    distance_mid_um=frame.distance_mid_um,signed_intra_density_um_inv=signed,
                    area_normalized_density_um_inv=area,positive_peak_normalized_intra=peak,
                    signed_area=signed_area,positive_peak_density_um_inv=positive_peak,
                    signed_valid=valid_intra,area_status=str(row.shape_status),
                    peak_status=peak_reason))
                outputs.append(d)
                metrics.append(dict(method=method,group=group,scope=scope,beta=float(row.beta),
                    beta_raw=float(row.beta_raw),signed_area=signed_area,
                    main_mass=float(row.main_mass),tail_mass=float(row.tail_mass),
                    tail_main_ratio=float(row.tail_main_ratio),negative_mass=float(row.negative_mass),
                    valid=valid_intra,shape_valid=str(row.shape_valid).lower() in ('true','1'),
                    shape_status=str(row.shape_status),positive_peak_density_um_inv=positive_peak,
                    peak_status=peak_reason,total_pair_count=int(row.total_pair_count),
                    background_ordered_nonzero_count=int(row.background_ordered_nonzero_count)))
    curves=pd.concat(outputs,ignore_index=True)
    scalars=pd.DataFrame(metrics)
    curves.to_csv(run/'data/comparison_curves.csv',index=False)
    scalars.to_csv(run/'data/comparison_curve_scalars.csv',index=False)
    differences=[]
    for group in ['On','Not-on']:
        for scope in scopes:
            a=scalars[(scalars.method=='A_default')&(scalars.group==group)&(scalars.scope==scope)].iloc[0]
            b=scalars[(scalars.method=='B_strong_union')&(scalars.group==group)&(scalars.scope==scope)].iloc[0]
            ca=curves[(curves.method=='A_default')&(curves.group==group)&(curves.scope==scope)]
            cb=curves[(curves.method=='B_strong_union')&(curves.group==group)&(curves.scope==scope)]
            np.testing.assert_array_equal(ca.distance_mid_um,cb.distance_mid_um)
            va=ca.area_normalized_density_um_inv.to_numpy(float)
            vb=cb.area_normalized_density_um_inv.to_numpy(float)
            available=bool(a.shape_valid and b.shape_valid and np.isfinite(va).all() and np.isfinite(vb).all())
            delta=vb-va
            index=int(np.argmax(abs(delta))) if available else None
            widths=(ca.distance_right_um-ca.distance_left_um).to_numpy(float)
            differences.append(dict(group=group,scope=scope,
                signed_area_default=float(a.signed_area),signed_area_hybrid=float(b.signed_area),
                signed_area_change=float(b.signed_area-a.signed_area),
                beta_default=float(a.beta),beta_hybrid=float(b.beta),beta_change=float(b.beta-a.beta),
                area_shape_difference_available=available,
                max_abs_area_normalized_density_difference=float(abs(delta[index])) if available else np.nan,
                max_difference_distance_um=float(ca.distance_mid_um.iloc[index]) if available else np.nan,
                integrated_abs_area_normalized_density_difference=float(np.sum(abs(delta)*widths)) if available else np.nan,
                default_shape_status=str(a.shape_status),hybrid_shape_status=str(b.shape_status),
                interpretation='Descriptive hybrid minus default; not a significance test or confidence interval'))
    pd.DataFrame(differences).to_csv(run/'data/curve_method_comparison_metrics.csv',index=False)
    dump(run/'data/comparison_curve_definition.json',dict(
       scopes=scopes,groups=['On','Not-on'],methods=['A_default','B_strong_union'],
       signed='Frozen signed Intra density, retain negatives, no smoothing.',
       area='Frozen signed Intra divided by its signed area over all300bins; original validity guards retained.',
       peak='Each group/scope signed Intra divided by its own maximum positive signed density over0-3um; unavailable if original Intra invalid/nonfinite or positive peak<=0.',
       units=dict(signed_intra_density_um_inv='1/um',area_normalized_density_um_inv='1/um',positive_peak_normalized_intra='dimensionless'),
       baseline_reused=True,normalization_is_science_not_presentation=True,formal_inference=False))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args()
    run=args.run.resolve();cfg=read(run/'config.json')
    if not (run/'MASK_ASSIGNMENT_COMPLETE.json').is_file():
        raise RuntimeError('Mask/assignment completion marker is missing; no upstream steps are launched.')
    if (run/'TARDIS_COMPLETE.json').is_file():
        print('TARDIS_ALREADY_COMPLETE',flush=True);return
    baseline=Path(cfg['source_output']).resolve()
    package=Path(cfg['pipeline_package']).resolve()
    out=run/'data/tardis';out.mkdir(parents=True,exist_ok=True)
    cache=run/'runtime';cache.mkdir(exist_ok=True)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(cache/'matplotlib'),
                      NUMBA_CACHE_DIR=str(cache/'numba'),TMPDIR=str(cache/'tmp'))
    (cache/'tmp').mkdir(parents=True,exist_ok=True)
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        os.environ[key]='1'
    # -B forbids bytecode writes; reuse read-only installed cache in active environment.
    # Scientific inputs, outputs and any writable runtime caches stay on My Book.
    frozen=freeze(package,run)
    sys.path.insert(0,str(frozen/'src'))
    print('HYBRID_TARDIS_IMPORT: NumPy',flush=True)
    import numpy as np
    print('HYBRID_TARDIS_IMPORT: pandas',flush=True)
    import pandas as pd
    print('HYBRID_TARDIS_IMPORT: frozen modular stages; NumPy='+np.__version__+' pandas='+pd.__version__+' python='+sys.executable,flush=True)
    from trak_pipeline.types import Sample,Artifact,StageContext
    from trak_pipeline.stages import distances,tardis
    start=time.perf_counter()
    original=baseline/'data/localizations.csv.gz'
    labels=run/'data/assignment_labels.csv.gz'
    fullmask=run/'masks/clean_full_mask.npy'
    protected=[original,baseline/'classification_summary.json',baseline/'data/tardis/metadata.json',
               baseline/'masks/clean_full_mask.npy',baseline/'masks/roi_mask.npy']
    before={str(p):descriptor(p) for p in protected}
    c=dict(cfg['tardis'])
    c['shared_weights']=str(Path(cfg['source_release'])/'bootstrap_weights.npz')
    c['mask_label']='default Nellie OR per-frame raw-C1 median+6MADscale; support50>=10/post-min32/holes<64'
    c['group_rule']='New saved UID labels; original k=1 own position-SE ellipse contact; original QC/ROI/unknown rules; no probability filter or weighting'
    c['sample_name']=cfg['sample_name'];c['sample_id']=cfg['sample_id']
    if (c['frames'],c['tau_frames'],c['distance_max_um'],c['distance_bins'])!=(2500,1,3.,300):
        raise ValueError('Isolated test requires unchanged original TARDIS grid and tau')
    if c['bootstrap']['replicates']!=1000 or c['bootstrap']['block_lengths_frames']!=[10,25,50]:
        raise ValueError('Frozen1000-replicate L10/L25/L50 scheme was changed')
    meta=read(baseline/'job.json')
    metadata={k:meta.get(k) for k in ['sample_name','molecule','genotype','BR','cell','acquisition_date']}
    metadata['run_id']=cfg['run_id']
    sample=Sample(cfg['sample_id'],Path(cfg['cell_directory']),metadata,
        Path(cfg['source_C1']),Path(cfg['cell_directory'])/Path(meta['C2_file'].replace('\\','/')).name,
        Path(cfg['source_ROI']))
    label_sha=fullsha(labels)
    fit_id=identity('legacy_verified_original_fit_v1',{},dict(source=before[str(original)]))
    label_id=identity('ellipse_k1_closed_pixels_v1',dict(k=1,tolerance=1e-12,mask_method=cfg['method_version']),
         dict(fit_id=fit_id,label_SHA256=label_sha,mask=descriptor(fullmask)))
    fits=Artifact(fit_id,'fit',dict(table=str(original)),dict(read_only=True))
    assignment=Artifact(label_id,'assignment',dict(labels=str(labels)),dict(UID_key='source_detection_uid'))
    config=dict(acquisition=dict(shape_yx=[684,428],frames=2500,pixel_size_nm=117.,frame_interval_s=.02),
                methods=dict(tardis=dict(parameters=c),bootstrap=dict(parameters=c['bootstrap'])))
    common=dict(config=config,sample=sample,parameters={},package_root=frozen)
    t0=time.perf_counter()
    print('HYBRID_TARDIS_STAGE: UID-bound distance counts',flush=True)
    dctx=StageContext(**common,inputs=dict(fit=fits,assignment=assignment),output_dir=out/'pair_histograms')
    dres=distances.execute(dctx)
    d_id=identity('tau1_uid_pair_counts_v1',dict(tau=1,edges_um=np.linspace(0,3,301).tolist(),
         total_rule=c['total_rule'],background_rule=c['background_rule']),dict(fit=fit_id,assignment=label_id))
    dart=Artifact(d_id,'distances',{k:str(v) for k,v in dres.files.items()},dres.metadata)
    count_seconds=time.perf_counter()-t0
    hist=distances.kernel()
    selected=pd.read_csv(dres.files['selection'],float_precision='round_trip',keep_default_na=False)
    selected.frame=selected.frame.astype(int)
    selected.fit_x_px=selected.fit_x_px.astype(float);selected.fit_y_px=selected.fit_y_px.astype(float)
    with np.load(dres.files['counts'],allow_pickle=False) as saved:
        h={k:saved[k].copy() for k in saved.files}
    checked=hist.independent_histogram_check(selected,h,c,cfg['sample_frames'])
    # Relabeling cannot alter the selected all-group point set or pair counts.
    old_histogram=baseline/'data/tardis/pair_histograms/per_frame_raw_counts.npz'
    with np.load(old_histogram,allow_pickle=False) as old_hist:
        for key in ('edges_um','all_total','all_background'):
            np.testing.assert_array_equal(h[key],old_hist[key])
    all_selected_pair_counts_invariant=True
    labels_table=pd.read_csv(labels,keep_default_na=False)
    old=pd.read_csv(original,usecols=['source_detection_uid','frame','qc_unflagged_model_fit','fitted_center_inside_ROI'],keep_default_na=False)
    if len(labels_table)!=len(old):raise AssertionError('Label/master candidate count changed')
    original_selected=distances.boolean(old.qc_unflagged_model_fit)&distances.boolean(old.fitted_center_inside_ROI)
    if len(selected)!=int(original_selected.sum()):raise AssertionError('Original selection count changed')
    frame_saved=pd.read_csv(run/'data/per_frame_counts.csv')
    for group,col in [('On','On'),('Not-on','Not_on'),('unknown','unknown')]:
        counts=selected[selected.tardis_group==group].groupby('frame').size().reindex(range(2500),fill_value=0)
        np.testing.assert_array_equal(counts.to_numpy(),frame_saved[col].to_numpy())
    print('HYBRID_TARDIS_STAGE: frozen decomposition, full and ten scopes',flush=True)
    t0=time.perf_counter()
    decctx=StageContext(**common,inputs=dict(distances=dart),output_dir=out)
    decres=tardis.execute_decomposition(decctx)
    dec_id=identity('frozen_signed_tardis_v1',dict(beta_fit=c['background_fit_um'],shape_rule=c['shape_rule'],
         minimum_total_tail_pairs=c['minimum_total_tail_pairs'],
         minimum_background_tail_ordered_pairs=c['minimum_background_tail_ordered_pairs']),
         dict(distances=d_id))
    decart=Artifact(dec_id,'decomposition',{k:str(v) for k,v in decres.files.items()},decres.metadata)
    dump(out/'decomposition_metadata.json',decres.metadata)
    decart.files['metadata']=str(out/'decomposition_metadata.json')
    decomposition_seconds=time.perf_counter()-t0
    print('HYBRID_TARDIS_STAGE:1000 shared paired temporal bootstrap L10/L25/L50',flush=True)
    t0=time.perf_counter()
    bsctx=StageContext(**common,inputs=dict(distances=dart,decomposition=decart),output_dir=out)
    bsres=tardis.execute_bootstrap(bsctx)
    bootstrap_seconds=time.perf_counter()-t0
    dump(out/'bootstrap_stage_metadata.json',bsres.metadata)
    weights=Path(cfg['source_release'])/'bootstrap_weights.npz'
    with np.load(out/'shared_weights.npz',allow_pickle=False) as actual,np.load(weights,allow_pickle=False) as ref:
        for length in c['bootstrap']['block_lengths_frames']:
            np.testing.assert_array_equal(actual[f'L{length}'],ref[f'L{length}'])
    print('HYBRID_TARDIS_STAGE: save signed, area and positive-peak comparison curves',flush=True)
    export_comparison(run,baseline,out,c,np,pd)
    unchanged={str(p):descriptor(p)==before[str(p)] for p in protected}
    if not all(unchanged.values()):raise AssertionError('Read-only baseline source changed')
    # Keep summary points explicitly, without copying any original fit columns.
    points=selected.groupby('tardis_group').size().to_dict()
    scopes=hist.scopes(c)
    summary=pd.read_csv(out/'per_scope_summary.csv',float_precision='round_trip')
    summary.insert(2,'points',[int(((selected.tardis_group==row.group)&selected.frame.between(int(row.start_frame),int(row.stop_exclusive)-1)).sum()) for row in summary.itertuples()])
    summary.to_csv(out/'per_scope_summary.csv',index=False)
    inputs=dict(fit_master=descriptor(original),UID_labels=descriptor(labels),full_mask=descriptor(fullmask),
        original_weights=descriptor(weights),generated_weights=descriptor(out/'shared_weights.npz'),
        shared_weight_arrays_exactly_original=True,source_master_read_only=True,original_QC_and_coordinates_reused=True)
    report=dict(sample_id=cfg['sample_id'],run_id=cfg['run_id'],method_version=cfg['method_version'],
       group_counts=points,selected_count=len(selected),original_candidate_count=len(old),
       original_QC_excluded=int((~original_selected).sum()),original_unknown_rules_retained=True,
       independent_histogram_group_frames_checked=int(checked),independent_frames=cfg['sample_frames'],
       source_unchanged=unchanged,
       scientific_calls=dict(detection=0,fitting=0,nellie=0,classification=0,
         distance_counts=1,decomposition=1,bootstrap=1,comparison_normalization=1),
       timing_seconds=dict(distance_counts=count_seconds,decomposition=decomposition_seconds,
         bootstrap=bootstrap_seconds,total=time.perf_counter()-start),
       metadata=dict(registration='identity / independently unverified',calibration='manufacturer_nominal',
         intervals='Exploratory pointwise2.5-97.5percentile ranges; no simultaneous bands or hypothesis tests',
         segmentation_registration_uncertainty_propagated=False,block_join_pairs=False,negative_bins_retained=True),
       all_selected_pair_counts_invariant=all_selected_pair_counts_invariant,
       environment=dict(python_executable=sys.executable,python_version=sys.version,
         numpy=np.__version__,pandas=pd.__version__,
         scipy=__import__('importlib.metadata',fromlist=['version']).version('scipy')),
       inputs=inputs,config_rules=c,source_audit=run/'source/tardis_source_manifest.json',
       completed_utc=utc())
    dump(out/'metadata.json',report)
    dump(out/'COMPLETE.json',dict(all_passed=True,numerical_complete=True,selected_count=len(selected),
         paired_replicates=1000,density_banks_retained=False,original_QC_and_master_unchanged=True))
    dump(run/'TARDIS_COMPLETE.json',report)
    print('HYBRID_TARDIS_COMPLETE',json.dumps(dict(group_counts=points,total_seconds=report['timing_seconds']['total'])),flush=True)

if __name__=='__main__':main()
