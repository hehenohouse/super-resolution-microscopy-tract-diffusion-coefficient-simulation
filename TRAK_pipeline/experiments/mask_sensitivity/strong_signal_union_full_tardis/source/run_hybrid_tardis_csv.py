#!/usr/bin/env python3
"""Unchanged frozen NumPy TARDIS kernels with standard-library CSV I/O.

No pandas import. The CSVFrame adapter only writes tables; scientific numerical
function bodies, bootstrap normalization and original shared weights are reused.
"""
from __future__ import annotations
import argparse,ast,csv,gzip,hashlib,importlib.util,json,math,os,sys,time,types,uuid
from pathlib import Path
from datetime import datetime,timezone

def utc():return datetime.now(timezone.utc).isoformat()
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,Path):return str(x)
    if hasattr(x,'tolist'):return safe(x.tolist())
    if isinstance(x,float) and not math.isfinite(x):return None
    return x
def dump(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:t.write_text(json.dumps(safe(x),indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(t,path)
    finally:t.unlink(missing_ok=True)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def csvrows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def descriptor(path):
    path=Path(path);s=path.stat();h=hashlib.sha256()
    with path.open('rb') as f:
        if s.st_size<=2*1024*1024:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
            kind='complete_small_file'
        else:
            h.update(f.read(65536));f.seek(max(0,s.st_size-65536));h.update(f.read(65536));kind='first_last_64KiB'
    return dict(path=str(path),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sampled_sha256=h.hexdigest(),fingerprint_kind=kind)
def truth(x):return str(x).lower() in ('true','1')
def num(x):return float(x) if str(x).strip() else float('nan')

class CSVFrame:
    """The exact table-writing interface required by frozen bootstrap()."""
    def __init__(self,values):
        if isinstance(values,dict):
            sizes=[len(v) for v in values.values() if hasattr(v,'__len__') and not isinstance(v,(str,bytes))]
            n=sizes[0] if sizes else 1
            if any(k!=n for k in sizes):raise ValueError('CSV columns have unequal lengths')
            self.rows=[{k:(v[i] if hasattr(v,'__len__') and not isinstance(v,(str,bytes)) else v) for k,v in values.items()} for i in range(n)]
            self.fields=list(values)
        else:
            self.rows=[dict(r) for r in values]
            self.fields=list(dict.fromkeys(k for r in self.rows for k in r))
    def __contains__(self,key):return key in self.fields
    def copy(self):return CSVFrame(self.rows)
    def insert(self,index,key,value):
        if key in self.fields:raise ValueError('Duplicate CSV column '+key)
        self.fields.insert(index,key)
        for r in self.rows:r[key]=value
    def assign(self,**values):
        result=self.copy()
        for k,v in values.items():
            if k not in result.fields:result.fields.append(k)
            for r in result.rows:r[k]=v
        return result
    def to_csv(self,path,index=False):
        if index:raise ValueError('CSVFrame never writes an implicit row index')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        t=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
        try:
            with t.open('w',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=self.fields);w.writeheader();w.writerows(self.rows)
            os.replace(t,path)
        finally:t.unlink(missing_ok=True)

def kernel(frozen,np):
    root=frozen/'src/trak_pipeline/compat/vendor'
    ns=dict(np=np,pd=types.SimpleNamespace(DataFrame=CSVFrame),GROUPS=['On','Not-on'])
    parts=[
      ('picasso_full_sensitivity/full_core.py',('decompose_counts','range_histograms','block_histograms','shared_resampling_plan')),
      ('picasso_roi_sensitivity/roi_core.py',('scopes','decompose_roi','bootstrap_roi','scalar_row','finite_ci','paired_difference','distribution_table')),
      ('mito_tardis/mito_core.py',('tail_diagnostics','paired_curve_summary'))]
    audit=[]
    for relative,names in parts:
        p=root/relative
        parsed=ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        nodes=[n for n in parsed.body if isinstance(n,ast.FunctionDef) and n.name in names]
        if set(n.name for n in nodes)!=set(names):raise ValueError('Missing frozen numerical functions '+relative)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
        audit.append(dict(source=str(p),functions=list(names),source_SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),bodies_changed=False))
    return types.SimpleNamespace(**ns),audit

def comparison(run,baseline,out,observed,c,np,mc):
    rows=[];scalars=[]
    scopes=mc.scopes(c);width=c['distance_max_um']/c['distance_bins']
    for method,root in [('A_default',baseline/'data/tardis'),('B_strong_union',out)]:
        summaries=csvrows(root/'per_scope_summary.csv')
        for group in mc.GROUPS:
            for scope,_,_ in scopes:
                matches=[r for r in summaries if r['group']==group and r['scope']==scope]
                if len(matches)!=1:raise ValueError('Missing/duplicate saved curve validity')
                s=matches[0]
                curve=csvrows(root/'distribution'/group/(scope+'.csv'))
                signed=np.array([num(r['intra_density_um_inv']) for r in curve])
                area=np.array([num(r['shape_density_um_inv']) for r in curve])
                mids=np.array([num(r['distance_mid_um']) for r in curve])
                widths=np.array([num(r['distance_right_um'])-num(r['distance_left_um']) for r in curve])
                signed_area=num(s['signed_area']);finite=bool(np.isfinite(signed).all())
                peak=num(np.max(signed)) if finite else np.nan
                valid=truth(s['valid']);peak_valid=valid and finite and peak>0
                peak_norm=signed/peak if peak_valid else np.full(len(signed),np.nan)
                peak_status='available_positive_peak' if peak_valid else ('unavailable_original_intra' if not valid else 'unavailable_nonfinite_or_nonpositive_peak')
                if finite and not np.isclose(num(np.sum(signed*widths)),signed_area,rtol=1e-11,atol=1e-12):
                    raise ValueError('Saved density/signed area disagree')
                for j,r in enumerate(curve):
                    rows.append(dict(method=method,group=group,scope=scope,
                        distance_left_um=num(r['distance_left_um']),distance_right_um=num(r['distance_right_um']),
                        distance_mid_um=num(r['distance_mid_um']),signed_intra_density_um_inv=num(signed[j]),
                        area_normalized_density_um_inv=num(area[j]),positive_peak_normalized_intra=num(peak_norm[j]),
                        signed_area=signed_area,positive_peak_density_um_inv=peak,signed_valid=valid,
                        area_status=s['shape_status'],peak_status=peak_status))
                scalars.append(dict(method=method,group=group,scope=scope,beta=num(s['beta']),beta_raw=num(s['beta_raw']),
                    signed_area=signed_area,main_mass=num(s['main_mass']),tail_mass=num(s['tail_mass']),
                    tail_main_ratio=num(s['tail_main_ratio']),negative_mass=num(s['negative_mass']),
                    valid=valid,shape_valid=truth(s['shape_valid']),shape_status=s['shape_status'],
                    positive_peak_density_um_inv=peak,peak_status=peak_status,
                    total_pair_count=int(s['total_pair_count']),background_ordered_nonzero_count=int(s['background_ordered_nonzero_count'])))
    CSVFrame(rows).to_csv(run/'data/comparison_curves.csv')
    CSVFrame(scalars).to_csv(run/'data/comparison_curve_scalars.csv')
    differences=[]
    for group in mc.GROUPS:
        for scope,_,_ in scopes:
            a=next(r for r in scalars if r['method']=='A_default' and r['group']==group and r['scope']==scope)
            b=next(r for r in scalars if r['method']=='B_strong_union' and r['group']==group and r['scope']==scope)
            ca=[r for r in rows if r['method']=='A_default' and r['group']==group and r['scope']==scope]
            cb=[r for r in rows if r['method']=='B_strong_union' and r['group']==group and r['scope']==scope]
            np.testing.assert_array_equal([r['distance_mid_um'] for r in ca],[r['distance_mid_um'] for r in cb])
            va=np.array([r['area_normalized_density_um_inv'] for r in ca]);vb=np.array([r['area_normalized_density_um_inv'] for r in cb])
            good=bool(a['shape_valid'] and b['shape_valid'] and np.isfinite(va).all() and np.isfinite(vb).all())
            delta=vb-va;i=int(np.argmax(abs(delta))) if good else None
            widths=np.array([r['distance_right_um']-r['distance_left_um'] for r in ca])
            differences.append(dict(group=group,scope=scope,signed_area_default=a['signed_area'],signed_area_hybrid=b['signed_area'],
                signed_area_change=b['signed_area']-a['signed_area'],beta_default=a['beta'],beta_hybrid=b['beta'],beta_change=b['beta']-a['beta'],
                area_shape_difference_available=good,max_abs_area_normalized_density_difference=num(abs(delta[i])) if good else np.nan,
                max_difference_distance_um=ca[i]['distance_mid_um'] if good else np.nan,
                integrated_abs_area_normalized_density_difference=num(np.sum(abs(delta)*widths)) if good else np.nan,
                default_shape_status=a['shape_status'],hybrid_shape_status=b['shape_status'],
                interpretation='Descriptive hybrid minus default; not a significance test or confidence interval'))
    CSVFrame(differences).to_csv(run/'data/curve_method_comparison_metrics.csv')
    dump(run/'data/comparison_curve_definition.json',dict(scopes=[s[0] for s in scopes],groups=mc.GROUPS,
        methods=['A_default','B_strong_union'],signed='Frozen signed Intra density; negative bins retained, no smoothing.',
        area='Signed Intra divided by its signed area over all300bins; original support and denominator guards.',
        peak='Each signed group/scope curve divided by its own positive maximum over0-3um; invalid originals remain unavailable.',
        units=dict(signed_intra_density_um_inv='1/um',area_normalized_density_um_inv='1/um',positive_peak_normalized_intra='dimensionless'),
        baseline_reused=True,normalization_is_science_not_presentation=True,formal_inference=False))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);args=ap.parse_args()
    run=args.run.resolve();cfg=read(run/'config.json')
    if not (run/'COUNTS_COMPLETE.json').is_file():raise RuntimeError('Saved UID-bound distance counts are not complete; upstream is never launched.')
    if (run/'TARDIS_COMPLETE.json').is_file():print('TARDIS_ALREADY_COMPLETE',flush=True);return
    runtime=run/'runtime';(runtime/'tmp').mkdir(parents=True,exist_ok=True)
    for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(runtime/'tmp'),MPLCONFIGDIR=str(runtime/'matplotlib'),NUMBA_CACHE_DIR=str(runtime/'numba'))
    print('HYBRID_TARDIS_RUNTIME:',sys.executable,'; NumPy-only frozen kernels; standard CSV I/O; no pandas',flush=True)
    import numpy as np
    started=time.perf_counter();baseline=Path(cfg['source_output']);out=run/'data/tardis';out.mkdir(parents=True,exist_ok=True)
    frozen=run/'source/frozen_tardis';mc,audit=kernel(frozen,np)
    spec=importlib.util.spec_from_file_location('_frozen_bootstrap_csv',frozen/'src/trak_pipeline/compat/analysis_stage.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.np=np;helper.pd=types.SimpleNamespace(DataFrame=CSVFrame);helper.mc=mc;helper.dump=dump
    c=dict(cfg['tardis']);c['mask_label']='Default Nellie OR raw-C1 median+6MADscale; support50>=10/postmin32/holes<64'
    c.update(classification_label_field='pointwise_label',assignment_label_artifact='data/assignment_labels.csv.gz',
        assignment_uid_key='source_detection_uid',assignment_label_artifact_path_basis='relative_to_isolated_test_run')
    c['group_rule']='New UID labels; original1SE ellipse, original QC/ROI/unknown policy; no weighting'
    c['sample_name']=cfg['sample_name'];c['sample_id']=cfg['sample_id']
    weights=Path(cfg['source_release'])/'bootstrap_weights.npz';c['shared_weights']=str(weights)
    if (c['frames'],c['tau_frames'],c['distance_bins'],c['distance_max_um'])!=(2500,1,300,3.):raise ValueError('Original TARDIS grid/tau changed')
    if c['bootstrap']['replicates']!=1000 or c['bootstrap']['block_lengths_frames']!=[10,25,50]:raise ValueError('Original shared bootstrap plan changed')
    master=baseline/'data/localizations.csv.gz'
    protected=[master,baseline/'data/tardis/metadata.json',baseline/'masks/clean_full_mask.npy']
    before={str(p):descriptor(p) for p in protected}
    counts=out/'pair_histograms/per_frame_raw_counts.npz'
    with np.load(counts,allow_pickle=False) as saved:h={k:saved[k].copy() for k in saved.files}
    np.testing.assert_array_equal(h['edges_um'],np.linspace(0,3,301))
    with np.load(baseline/'data/tardis/pair_histograms/per_frame_raw_counts.npz',allow_pickle=False) as old:
        for key in ['all_total','all_background','edges_um']:np.testing.assert_array_equal(h[key],old[key])
    selected_rows=[]
    selection_path=out/'pair_histograms/selected_uid_coordinates_labels.csv.gz'
    with gzip.open(selection_path,'rt',encoding='utf-8',newline='') as f:selected_rows=list(csv.DictReader(f))
    point_frames={g:np.array([int(r['frame']) for r in selected_rows if r['tardis_group']==g]) for g in mc.GROUPS}
    group_counts={g:int(len(v)) for g,v in point_frames.items()}
    summary=[];observed={};flat={};t0=time.perf_counter()
    print('HYBRID_TARDIS_STAGE: frozen decomposition full and ten scopes',flush=True)
    for g,group in enumerate(mc.GROUPS):
        (out/'distribution'/group).mkdir(parents=True,exist_ok=True)
        for scope,start,stop in mc.scopes(c):
            tc,bc=mc.range_histograms(h['total'][g],h['background'][g],start,stop)
            d=mc.decompose_roi(tc,bc,c);observed[group,scope]=d
            flat.update({group+'__'+scope+'__'+k:v for k,v in d.items()})
            mc.distribution_table(tc,bc,d,c).assign(group=group,scope=scope).to_csv(out/'distribution'/group/(scope+'.csv'))
            summary.append(dict(group=group,scope=scope,start_frame=start,stop_exclusive=stop,
                points=int(((point_frames[group]>=start)&(point_frames[group]<stop)).sum()),
                total_pair_count=int(tc.sum()),background_ordered_nonzero_count=int(bc.sum()),
                background_exact_zero_removed=int(h['background_zero_removed'][g,start:stop].sum()),
                **mc.scalar_row(d),**mc.tail_diagnostics(tc,bc,h['total'][g],h['background'][g],start,stop,c)))
    np.savez_compressed(out/'observed_decomposition.npz',**flat)
    CSVFrame(summary).to_csv(out/'per_scope_summary.csv')
    decomposition_seconds=time.perf_counter()-t0
    (out/'bootstrap').mkdir(exist_ok=True)
    plans={};maps={}
    with np.load(weights,allow_pickle=False) as source:
        for length in c['bootstrap']['block_lengths_frames']:
            plans[f'L{length}']=source[f'L{length}'].copy()
            np.testing.assert_array_equal(plans[f'L{length}'],mc.shared_resampling_plan(c,length))
            starts=np.arange(0,c['frames'],length,dtype=np.int64)
            src=np.concatenate([np.arange(s,s+length-1) for s in starts])
            maps[f'L{length}__block_starts']=starts;maps[f'L{length}__source_frame']=src
            maps[f'L{length}__target_frame']=src+1;maps[f'L{length}__stratum']=starts//c['segment_frames']
            maps[f'L{length}__block_id']=np.repeat(np.arange(len(starts)),length-1)
            maps[f'L{length}__endpoint_valid']=h.get('endpoint_valid',np.r_[np.ones(2499,bool),False])[src]
    local_weights=out/'shared_weights.npz';np.savez_compressed(local_weights,**plans,replicate_id=np.arange(1000))
    np.savez_compressed(out/'resampling_units.npz',**maps)
    basejob=read(baseline/'job.json');job={k:basejob.get(k) for k in ['sample_id','molecule','genotype','BR','cell','acquisition_date']}
    job.update(run_id=cfg['run_id'],sample_name=cfg['sample_name'])
    print('HYBRID_TARDIS_STAGE: original1000 shared paired bootstrap L10/L25/L50',flush=True)
    t0=time.perf_counter();helper.bootstrap(out,c,h,observed,local_weights,job);bootstrap_seconds=time.perf_counter()-t0
    print('HYBRID_TARDIS_STAGE: save signed, area, positive-peak comparison derivatives',flush=True)
    comparison(run,baseline,out,observed,c,np,mc)
    unchanged={str(p):descriptor(p)==before[str(p)] for p in protected}
    if not all(unchanged.values()):raise AssertionError('Protected baseline input changed')
    report=dict(sample_id=cfg['sample_id'],run_id=cfg['run_id'],method_version=cfg['method_version'],
       group_counts=group_counts,selected_count=len(selected_rows),all_selected_pair_counts_invariant=True,
       original_QC_and_coordinates_reused=True,source_unchanged=unchanged,config_rules=c,
       inputs=dict(master=descriptor(master),UID_labels=descriptor(run/'data/assignment_labels.csv.gz'),
         counts=descriptor(counts),original_weights=descriptor(weights),shared_weights=descriptor(local_weights),
         shared_weight_arrays_exactly_original=True),source_audit=audit,
       environment=dict(python_executable=sys.executable,python_version=sys.version,numpy=np.__version__,
         pandas_imported=False,CSV_adapter='Writing only; no numerical or statistical operations',
         library_reason='Existing local pandas package reads stalled; use unchanged frozen NumPy kernels and stdlib CSV.'),
       scientific_calls=dict(detection=0,fitting=0,nellie=0,classification=0,distance_counts=0,decomposition=1,bootstrap=1,comparison_normalization=1),
       timing_seconds=dict(decomposition=decomposition_seconds,bootstrap=bootstrap_seconds,total=time.perf_counter()-started),
       registration='identity / independently unverified',calibration='manufacturer_nominal',
       intervals='Exploratory pointwise2.5-97.5percentile ranges, conditional on labels/masks; no formal significance test',
       full_bootstrap_curves_persisted=False,negative_bins_retained=True,block_join_pairs=False,
       completed_utc=utc())
    dump(out/'metadata.json',report)
    dump(out/'COMPLETE.json',dict(numerical_complete=True,paired_replicates=1000,density_banks_retained=False))
    dump(run/'TARDIS_COMPLETE.json',report)
    print('HYBRID_TARDIS_COMPLETE',json.dumps(dict(group_counts=group_counts,total_seconds=report['timing_seconds']['total'])),flush=True)
if __name__=='__main__':main()
