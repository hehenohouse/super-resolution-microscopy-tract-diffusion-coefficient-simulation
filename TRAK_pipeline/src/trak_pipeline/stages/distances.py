"""UID-bound raw pair counts. Science only: no plotting or upstream execution."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from ..types import StageResult
from ..kernels import science as science_paths

VENDOR=Path(__file__).resolve().parents[1]/'compat'/'vendor'
GROUPS=('On','Not-on')
UID_FIELDS=('source_detection_uid','detection_uid','uid')


def dump(path,value):
    from ..io import atomic_json
    atomic_json(path,value)


def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def kernel():
    """Load exact frozen numerical function ASTs without localization imports.

    Function bodies are unchanged; importing full_core itself would import the
    Picasso detector/fitter and GUI plotting dependencies into a count worker.
    Frozen-source fingerprints are retained by the release audit separately.
    """
    global _KERNEL
    if '_KERNEL' in globals():return _KERNEL
    import ast,types
    from scipy.spatial.distance import cdist
    ns=dict(np=np,pd=pd,cdist=cdist,GROUPS=list(GROUPS))
    parts=[('picasso_full_sensitivity/full_core.py',('decompose_counts','range_histograms','block_histograms','shared_resampling_plan')),
           ('picasso_roi_sensitivity/roi_core.py',('scopes','decompose_roi','bootstrap_roi','scalar_row','finite_ci','paired_difference','distribution_table')),
           ('mito_tardis/mito_core.py',('histogram','arrays_by_group','independent_histogram_check','tail_diagnostics','paired_curve_summary')),
           ('mito_tardis/summarize_intra.py',('summarize',)),
           ('mito_tardis/postmin32_extras.py',('_statistics',))]
    for relative,names in parts:
        path=VENDOR/relative;parsed=ast.parse(path.read_text(encoding='utf-8'),filename=str(path))
        nodes=[node for node in parsed.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names]
        if {node.name for node in nodes}!=set(names):raise ValueError('Frozen numerical API changed: '+relative)
        if relative.endswith('postmin32_extras.py'):
            nodes=[node for node in parsed.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STAT_FIELDS' for t in node.targets)]+nodes
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    _KERNEL=types.SimpleNamespace(**ns)
    return _KERNEL


def rules(ctx):
    c=load_json(VENDOR.parent/'tardis_reference.json')
    c={k:v for k,v in c.items() if not k.startswith(('_','expected_'))}
    c.update(ctx.config.get('acquisition',{}))
    for name in ('distances','decomposition','tardis','bootstrap'):
        value=ctx.config.get('methods',{}).get(name,{})
        value=value.get('parameters',value)
        if name=='bootstrap':
            c['bootstrap']=dict(c['bootstrap'],**value);continue
        c.update({k:v for k,v in value.items() if k not in ('method','method_id','version','method_version')})
    c.update({k:v for k,v in ctx.parameters.items() if not k.startswith('_') and k not in ('method','method_id','version','method_version')})
    c['frames']=int(c.get('frames',c.get('frame_count',2500)))
    c['pixel_size_nm']=float(c.get('pixel_size_nm',117))
    c['frame_interval_s']=float(c.get('frame_interval_s',.02))
    c['tau_frames']=int(c.get('tau_frames',1))
    if c['tau_frames']!=1: raise ValueError('This method supports tau_frames=1 only; another tau requires a new method version')
    if 'edges_um' in c:c['bin_edges_um']=c['edges_um']
    if 'bin_edges_um' in c:
        edges=np.asarray(c['bin_edges_um'],float)
        if len(edges)<2 or edges[0]!=0 or not np.isfinite(edges).all() or not (np.diff(edges)>0).all(): raise ValueError('Invalid bin edges')
        if not np.allclose(np.diff(edges),np.diff(edges)[0],rtol=0,atol=1e-12): raise ValueError('Frozen decomposition requires uniform bins; new edges require a uniform-bin method')
        c['distance_bins']=len(edges)-1;c['distance_max_um']=float(edges[-1])
    c['sample_name']=ctx.sample.metadata.get('sample_name',ctx.sample_id)
    return c


def uid_column(table):
    names=[key for key in UID_FIELDS if key in table]
    if not names: raise ValueError('Stable persistent detection UID is missing; row numbers and coordinates are not substitutes')
    key=names[0]
    if table[key].isna().any() or table[key].astype(str).eq('').any() or not table[key].is_unique: raise ValueError('Detection UIDs must be present and unique')
    return key


def boolean(series):
    if series.dtype==bool: return series
    text=series.astype(str).str.lower()
    if not text.isin(['true','false','1','0']).all(): raise ValueError('Unrecognized saved boolean')
    return text.isin(['true','1'])


def joined_table(ctx):
    table=pd.read_csv(ctx.file('fit','table'),float_precision='round_trip',keep_default_na=False,dtype={name:str for name in UID_FIELDS})
    uid=uid_column(table)
    labels=pd.read_csv(ctx.file('assignment','labels'),keep_default_na=False,dtype={name:str for name in UID_FIELDS})
    labeluid=uid_column(labels)
    field=next((x for x in ('label','pointwise_label','postmin32_pointwise_label','tardis_group','assignment_label') if x in labels),None)
    if field is None: raise ValueError('UID label artifact has no declared spatial-label field')
    if set(labels[labeluid].astype(str))!=set(table[uid].astype(str)): raise ValueError('Label UID set does not correspond one-to-one to fit table')
    mapped=labels.assign(_uid=labels[labeluid].astype(str)).set_index('_uid')[field]
    table['tardis_group']=table[uid].astype(str).map(mapped).replace({'mito-compatible':'On','non-mito-compatible':'Not-on'})
    if 'spatial_selected' in table: keep=boolean(table.spatial_selected)
    elif {'qc_unflagged_model_fit','fitted_center_inside_ROI'}<=set(table): keep=boolean(table.qc_unflagged_model_fit)&boolean(table.fitted_center_inside_ROI)
    else: raise ValueError('Original QC and refined-center ROI selection fields are required')
    if {'qc_unflagged_model_fit','fitted_center_inside_ROI'}<=set(table):
        original=boolean(table.qc_unflagged_model_fit)&boolean(table.fitted_center_inside_ROI)
        if not np.array_equal(keep,original):raise ValueError('Saved selection disagrees with original QC and refined-center ROI')
    if 'spatial_selected' in labels:
        expected=labels.assign(_uid=labels[labeluid].astype(str)).set_index('_uid').spatial_selected
        if not np.array_equal(boolean(table[uid].astype(str).map(expected)),keep):raise ValueError('Assignment artifact changed original QC selection')
    selected=table.loc[keep].copy()
    if not selected.tardis_group.isin(['On','Not-on','unknown']).all(): raise ValueError('Selected labels must be On, Not-on or unknown')
    for name in ('frame','fit_x_px','fit_y_px','x_se_nm','y_se_nm'):
        if name not in selected: raise ValueError('Missing explicit localization field '+name)
        selected[name]=pd.to_numeric(selected[name],errors='raise')
    if not np.isfinite(selected[['frame','fit_x_px','fit_y_px','x_se_nm','y_se_nm']].to_numpy(float)).all(): raise ValueError('Selected fit data must be finite')
    if (selected[['x_se_nm','y_se_nm']]<=0).any().any(): raise ValueError('Position errors must be positive nominal per-axis SE in nm')
    return table,selected,keep


def execute(ctx):
    c=rules(ctx); table,selected,keep=joined_table(ctx);n=c['frames']
    if not selected.frame.between(0,n-1).all() or not np.equal(selected.frame,selected.frame.astype(int)).all(): raise ValueError('Invalid original camera frame')
    # Explicit edges are used for counting; no sorting/rounding of native centers.
    edges=np.asarray(c.get('bin_edges_um',np.linspace(0,c['distance_max_um'],c['distance_bins']+1)),float)
    mc=kernel(); coords=mc.arrays_by_group(selected,c);k=len(edges)-1
    hist={key:np.zeros((2,n,k),np.int64) for key in ('total','background','cross')}
    hist.update({key:np.zeros((n,k),np.int64) for key in ('same_cross_background','all_total','all_background')})
    zero=np.zeros((2,n),np.int64)
    frame_valid=np.ones(n,bool)
    supplied=ctx.parameters.get('valid_original_frames')
    if supplied is not None:
        frame_valid[:]=False;frame_valid[np.asarray(supplied,int)]=True
    endpoint_valid=frame_valid & np.r_[frame_valid[1:],False]
    for f in range(n):
        if frame_valid[f]:
            for g,group in enumerate(GROUPS): hist['background'][g,f],zero[g,f]=mc.histogram(coords[group][f],coords[group][f],edges,True)
            hist['same_cross_background'][f]=2*mc.histogram(coords['On'][f],coords['Not-on'][f],edges,True)[0]
            both=np.concatenate([coords[g][f] for g in GROUPS]);hist['all_background'][f]=mc.histogram(both,both,edges,True)[0]
        if endpoint_valid[f]:
            for g,group in enumerate(GROUPS): hist['total'][g,f]=mc.histogram(coords[group][f],coords[group][f+1],edges)[0]
            hist['cross'][0,f]=mc.histogram(coords['On'][f],coords['Not-on'][f+1],edges)[0]
            hist['cross'][1,f]=mc.histogram(coords['Not-on'][f],coords['On'][f+1],edges)[0]
            hist['all_total'][f]=hist['total'][:,f].sum(0)+hist['cross'][:,f].sum(0)
        if (f+1)%250==0 or f+1==n: ctx.progress('Distance-count checkpoint',frame=f+1)
    hist.update(background_zero_removed=zero,edges_um=edges,original_frame=np.arange(n),source_frame=np.arange(n),target_frame=np.arange(n)+1,frame_valid=frame_valid,endpoint_valid=endpoint_valid,group_order=np.array(GROUPS))
    ctx.output_dir.mkdir(parents=True,exist_ok=True);path=ctx.output_dir/'per_frame_raw_counts.npz';np.savez_compressed(path,**hist)
    uid=uid_column(selected)
    selected[[uid,'frame','fit_x_px','fit_y_px','tardis_group']].to_csv(ctx.output_dir/'selected_uid_coordinates_labels.csv.gz',index=False,compression='gzip')
    counts=dict(selected=0,On=0,Not_on=0,unknown=0,QC_excluded=0)
    # Selection identity includes UID, frame, floating coordinate bytes and labels.
    rows=selected[[uid,'frame','fit_x_px','fit_y_px','tardis_group']].sort_values(uid).to_csv(index=False,float_format='%.17g').encode()
    metadata=dict(config_rules=c,selection_identity_sha256=hashlib.sha256(rows).hexdigest(),groups=GROUPS,raw_axes=['group','original_source_frame','distance_bin'],bootstrap_sufficient_units='Per-source-frame Total, per-frame ordered B, zero removals and original endpoint validity; never only pooled counts',cross_frame_exact_zero_kept=True,same_frame_all_exact_zero_removed=True,registration_status='identity / unverified',calibration_status='manufacturer_nominal')
    dump(ctx.output_dir/'metadata.json',metadata)
    return StageResult(dict(counts=path,metadata=ctx.output_dir/'metadata.json',selection=ctx.output_dir/'selected_uid_coordinates_labels.csv.gz'),metadata)

execute_distances=execute
