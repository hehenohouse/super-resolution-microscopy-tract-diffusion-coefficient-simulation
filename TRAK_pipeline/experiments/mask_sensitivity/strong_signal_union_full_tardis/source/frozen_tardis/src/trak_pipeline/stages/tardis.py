"""Separately cacheable decomposition and paired temporal bootstrap stages."""
from __future__ import annotations
from pathlib import Path
import importlib.util, shutil
import numpy as np
import pandas as pd
from ..types import StageResult
from .distances import dump,load_json,kernel,rules,VENDOR,GROUPS

FIELDS=('total','background','inter','intra','shape')
METRICS=('beta','signed_area','main_mass','tail_mass','tail_main_ratio')


def _hist(ctx):
    with np.load(ctx.file('distances','counts'),allow_pickle=False) as saved: return {k:saved[k].copy() for k in saved.files}


def execute_decomposition(ctx):
    c=rules(ctx);mc=kernel();h=_hist(ctx);out=ctx.output_dir;out.mkdir(parents=True,exist_ok=True)
    edges=h['edges_um'];width=np.diff(edges)
    if len(width)!=c['distance_bins'] or not np.allclose(width,width[0],rtol=0,atol=1e-12): raise ValueError('Counts and uniform decomposition bin definition disagree')
    if not np.allclose(edges,np.linspace(0,c['distance_max_um'],c['distance_bins']+1),rtol=0,atol=1e-12): raise ValueError('Decomposition bin edges differ from saved count edges')
    if c['frames']%c['segment_frames']: raise ValueError('Frozen decomposition scopes require complete segment_frames windows')
    all_observed={};summary=[];effects=[]
    for g,group in enumerate(GROUPS):
        directory=out/'distribution'/group;directory.mkdir(parents=True,exist_ok=True)
        for scope,start,stop in mc.scopes(c):
            tc,bc=mc.range_histograms(h['total'][g],h['background'][g],start,stop)
            d=mc.decompose_roi(tc,bc,c)
            for field,value in d.items(): all_observed[group+'__'+scope+'__'+field]=value
            mc.distribution_table(tc,bc,d,c).assign(group=group,scope=scope).to_csv(directory/(scope+'.csv'),index=False)
            row=dict(group=group,scope=scope,start_frame=start,stop_exclusive=stop,total_pair_count=int(tc.sum()),background_ordered_nonzero_count=int(bc.sum()),background_exact_zero_removed=int(h['background_zero_removed'][g,start:stop].sum()),**mc.scalar_row(d),**mc.tail_diagnostics(tc,bc,h['total'][g],h['background'][g],start,stop,c))
            summary.append(row)
    for scope,start,stop in mc.scopes(c):
        a=all_observed['On__'+scope+'__shape'][0];b=all_observed['Not-on__'+scope+'__shape'][0]
        difference=(a-b)/width
        effects.append(dict(scope=scope,shape_difference_available=bool(np.isfinite(difference).all()),max_abs_density_difference=float(np.max(abs(difference))),integrated_abs_signed_density_difference=float(np.sum(abs(difference)*width))))
    np.savez_compressed(out/'observed_decomposition.npz',**all_observed)
    pd.DataFrame(summary).to_csv(out/'per_scope_summary.csv',index=False)
    pd.DataFrame(effects).to_csv(out/'descriptive_shape_differences.csv',index=False)
    metadata=dict(config_rules=c,source_distance_artifact=ctx.input('distances').id,negative_bins_retained=True,formal_inference=False)
    dump(out/'metadata.json',metadata)
    return StageResult(dict(directory=out,metadata=out/'metadata.json',observed=out/'observed_decomposition.npz',summary=out/'per_scope_summary.csv'),metadata)


def _observed(ctx,c):
    artifact=ctx.input('decomposition');path=artifact.files.get('observed')
    if path and Path(path).is_file():
        with np.load(path,allow_pickle=False) as saved:
            return {(group,scope):{name.split('__',2)[2]:saved[name].copy() for name in saved.files if name.startswith(group+'__'+scope+'__')} for group in GROUPS for scope,_,_ in kernel().scopes(c)}
    directory=artifact.file('directory')
    summary=pd.read_csv(directory/'per_scope_summary.csv',float_precision='round_trip')
    observed={};width=c['distance_max_um']/c['distance_bins']
    for group in GROUPS:
        for scope,_,_ in kernel().scopes(c):
            curve=pd.read_csv(directory/'distribution'/group/(scope+'.csv'),float_precision='round_trip')
            rows=summary[(summary.group==group)&(summary.scope==scope)]
            if len(rows)!=1:raise ValueError('Saved observed decomposition validity/scalars missing')
            row=rows.iloc[0];d={field:curve[field+'_density_um_inv'].to_numpy(float)[None,:]*width for field in FIELDS}
            for field in (*METRICS,'beta_raw','negative_mass','valid','shape_valid','ratio_valid','tail_supported','shape_status','total_tail_count','background_tail_count','total_tail_nonempty_bins','background_tail_nonempty_bins'):
                if field not in row:raise ValueError('Saved observed decomposition scalar missing: '+field)
                d[field]=np.asarray([row[field]])
            observed[group,scope]=d
    return observed


def execute_bootstrap(ctx):
    c=rules(ctx);mc=kernel();h=_hist(ctx);observed=_observed(ctx,c);out=ctx.output_dir
    (out/'bootstrap').mkdir(parents=True,exist_ok=True)
    bs=dict(c['bootstrap']);bs.update(ctx.parameters.get('bootstrap',{}))
    for key in ('replicates','seed','block_lengths_frames','primary_block_frames','strata_frames'):
        if key in ctx.parameters: bs[key]=ctx.parameters[key]
    c['bootstrap']=bs
    if int(bs.get('strata_frames',c['segment_frames']))!=c['segment_frames']: raise ValueError('Bootstrap strata must match saved decomposition scopes')
    lengths=[int(v) for v in bs['block_lengths_frames']]
    for length in lengths:
        if length<=1 or c['segment_frames']%length or c['frames']%length: raise ValueError('Bootstrap block must divide each original stratum')
    # These source/target maps document true internal pairs explicitly. Weights
    # select whole source blocks; there is no concatenated pseudo-movie.
    plans={};maps={}
    for length in lengths:
        plans[f'L{length}']=mc.shared_resampling_plan(c,length)
        starts=np.arange(0,c['frames'],length,dtype=np.int64)
        source=np.concatenate([np.arange(s,s+length-1) for s in starts])
        maps[f'L{length}__block_starts']=starts;maps[f'L{length}__source_frame']=source;maps[f'L{length}__target_frame']=source+1
        maps[f'L{length}__stratum']=starts//c['segment_frames'];maps[f'L{length}__block_id']=np.repeat(np.arange(len(starts)),length-1)
        maps[f'L{length}__endpoint_valid']=h.get('endpoint_valid',np.r_[np.ones(c['frames']-1,bool),False])[source]
    weights=out/'shared_weights.npz';np.savez_compressed(weights,**plans,replicate_id=np.arange(bs['replicates']))
    np.savez_compressed(out/'resampling_units.npz',**maps)
    # Reuse precisely the frozen bootstrap algebra and compact persistence, not
    # analysis_stage.run (which would recompute counts and unrelated stages).
    spec=importlib.util.spec_from_file_location('_trak_frozen_bootstrap',VENDOR.parent/'analysis_stage.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    helper.np=np;helper.pd=pd;helper.mc=mc;helper.dump=dump
    job=dict(ctx.sample.metadata,sample_id=ctx.sample_id,sample_name=c['sample_name'],run_id=ctx.input('distances').id)
    helper.bootstrap(out,c,h,observed,weights,job)
    metadata=dict(config_rules=c,source_distance_artifact=ctx.input('distances').id,source_decomposition_artifact=ctx.input('decomposition').id,shared_weights=str(weights),resampling_units=str(out/'resampling_units.npz'),paired_original_replicate_ids=True,block_join_pairs=False,density_draws_retained=False,exploratory_only=True)
    dump(out/'metadata.json',metadata)
    return StageResult(dict(directory=out/'bootstrap',root=out,metadata=out/'metadata.json',weights=weights,units=out/'resampling_units.npz'),metadata)

execute=execute_decomposition
