"""Explicit hierarchy science, followed by a separate saved-data publisher.

summarize computes cohort membership, normalization and equal-weight statistics.
refresh reads persisted node payloads and prepared workbook cells only.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv,html,json,os,sys,hashlib,uuid
import numpy as np
import pandas as pd
from ..identity import digest
from ..io import replace as atomic_replace
from ..types import Sample
from ..kernels import science as science_paths,COMPAT
from .distances import dump,load_json,GROUPS

METRICS=('on_percentage','first_mask_ROI_area_um2','mean_ROI_area_um2')
COUNT_FIELDS=('On','Not_on','unknown','QC_excluded','selected','ROI_candidates')
VIEWS=('signed','area','peak')


def _portable(root,value):
    root=Path(root).resolve();text=str(value).replace('\\','/');path=Path(text)
    if path.is_absolute() and path.resolve().is_relative_to(root):return path.resolve()
    token='/'+root.name+'/'
    if token in text:text=text.rsplit(token,1)[1]
    elif path.is_absolute() or (len(text)>1 and text[1]==':'):raise ValueError('Summary path is outside data root: '+text)
    result=(root/text).resolve()
    if not result.is_relative_to(root):raise ValueError('Summary path escapes data root')
    return result


def _save_csv(frame,path,**options):
    path=Path(path);temporary=path.with_name('.'+uuid.uuid4().hex+'_'+path.name)
    try:frame.to_csv(temporary,index=False,**options);atomic_replace(temporary,path)
    finally:
        if temporary.exists():temporary.unlink()


def _plain(value):
    if isinstance(value,dict):return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_plain(v) for v in value]
    if hasattr(value,'tolist'):return _plain(value.tolist())
    if isinstance(value,Path):return str(value)
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def _stats(values,ncells=None,nbr=0):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],float);n=len(a)
    return dict(n=n,mean=float(a.mean()) if n else None,sd=float(a.std(ddof=1)) if n>1 else None,sem=float(a.std(ddof=1)/np.sqrt(n)) if n>1 else None,n_cells_valid=n if ncells is None else ncells,n_BRs_valid=nbr)


def _samples(samples,artifacts_by_sample):
    for item in samples:
        if isinstance(item,Sample): yield item,(artifacts_by_sample or {}).get(item.id,{})
        elif isinstance(item,dict) and isinstance(item.get('sample'),Sample):yield item['sample'],item.get('artifacts',(artifacts_by_sample or {}).get(item['sample'].id,{}))
        else:raise TypeError('summarize requires Sample objects or {sample: Sample, artifacts: mapping}')


def _summary_parameters(config):
    """Method v1 rejects scientific variants it cannot actually implement."""
    defaults=dict(cell_weighting='equal',condition_weighting='equal_replica',BR_weighting='equal',
        count_main_summary='pooled_per_cell',count_additional_mean='equal_replica',
        on_percentage_denominator='On+Not_on',intra_views=['signed','signed_area','peak'],
        intra_groups=['On','Not-on'],signed_area_min_abs=.01,minimum_signed_area_abs=.01,
        require_positive_peak=True,require_both_groups=True,area_order='normalize_BR_signed_mean',
        peak_within_BR='mean_cell_own_positive_peak_normalized',negative_bins_retained=True,SD_ddof=1)
    supplied=config.get('summary',{})
    if not isinstance(supplied,dict):raise ValueError('summary scientific parameters must be a mapping')
    for key,value in supplied.items():
        if key not in defaults:raise ValueError('Unsupported hierarchy summary scientific parameter: '+key)
        if value!=defaults[key]:raise ValueError('Unsupported hierarchy summary scientific setting '+key+'; method v1 requires '+str(defaults[key])+'; another method version is required')
    if list(config.get('conditions',['WT','TDK','MDK']))!=['WT','TDK','MDK']:raise ValueError('Hierarchy summary method v1 supports conditions WT, TDK, MDK in that order')
    return defaults


def _expected_inventory(config,completed,root):
    expected=config.get('_expected_samples')
    if expected is None:return list(completed)
    rows=[];seen=set()
    for item in expected:
        if isinstance(item,Sample):
            row=dict(item.metadata,sample_id=item.id,source_directory_relative=item.directory.relative_to(root).as_posix())
        elif isinstance(item,dict):
            row=dict(item)
            if isinstance(row.get('sample'),Sample):
                sample=row.pop('sample');row=dict(sample.metadata,**row,sample_id=sample.id,source_directory_relative=sample.directory.relative_to(root).as_posix())
        else:raise TypeError('Expected sample inventory must contain Sample objects or metadata mappings')
        sid=row.get('sample_id')
        if not sid or sid in seen:raise ValueError('Expected sample IDs must be present and unique')
        seen.add(sid);row.setdefault('genotype',row.get('condition'));row.setdefault('BR',row.get('replica'))
        location=row.get('source_directory_relative') or row.get('directory') or row.get('source_directory')
        if location is None:raise ValueError('Expected inventory lacks source_directory_relative for '+sid)
        row['source_directory_relative']=_portable(root,location).relative_to(root).as_posix()
        if row.get('genotype') not in ('WT','TDK','MDK') or not row.get('BR') or not row.get('cell'):raise ValueError('Expected inventory lacks condition/BR/cell identity for '+sid)
        rows.append(row)
    complete_ids={row['sample_id'] for row in completed}
    if not complete_ids<=seen:raise ValueError('Completed cells are outside the declared expected inventory')
    byid={row['sample_id']:row for row in rows}
    for row in completed:
        if any(row.get(key)!=byid[row['sample_id']].get(key) for key in ('genotype','BR','cell','source_directory_relative')):raise ValueError('Completed and expected cell identities disagree: '+row['sample_id'])
    return sorted(rows,key=lambda r:r['sample_id'])


def _load_cell(sample,artifacts,config):
    record=dict(sample.metadata,sample_id=sample.id,source_directory_relative=sample.directory.relative_to(Path(config['data_root'])).as_posix(),status='complete',molecule=sample.metadata.get('molecule',config.get('molecule','')))
    record.setdefault('genotype',record.get('condition'));record.setdefault('BR',record.get('replica'));record.setdefault('cell',sample.directory.name)
    old=sample.legacy_output;stat=artifacts.get('statistics');decomp=artifacts.get('decomposition');assignment=artifacts.get('assignment')
    if stat and stat.files.get('metadata') and Path(stat.files['metadata']).is_file():meta=load_json(stat.files['metadata'])
    else:meta={}
    if not meta.get('metrics') and stat and stat.metadata.get('metrics'):meta=dict(meta,metrics=stat.metadata['metrics'])
    if meta.get('metrics'):
        metric=meta['metrics'];record.update(metric['counts']);record.update({key:metric.get(key) for key in METRICS})
    elif old:
        path=Path(old)/'classification_summary.json'
        if not path.is_file() and assignment:path=assignment.file('summary')
        if not path.is_file():raise FileNotFoundError('Saved classification summary missing: '+sample.id)
        summary=load_json(path);counts=summary.get('counts',summary)
        for key in COUNT_FIELDS:
            source=key if key in counts else ('Not-on' if key=='Not_on' else key)
            if source not in counts:raise ValueError('Missing saved count '+key+' for '+sample.id)
            record[key]=int(counts[source])
        denom=record['On']+record['Not_on'];record['on_percentage']=100*record['On']/denom if denom else None
        maskstats=Path(old)/'masks/per_block_statistics.csv'
        if not maskstats.is_file():raise FileNotFoundError('Saved block area statistics missing: '+sample.id)
        areas=pd.read_csv(maskstats,float_precision='round_trip');key=next((k for k in ('mask_ROI_area_um2','ROI_area_um2','clean_ROI_area_um2','ROI_foreground_area_um2') if k in areas),None)
        if key is None:
            pxkey=next((k for k in ('mask_ROI_area_px2','mask_ROI_area_pixel2','ROI_area_pixel2','ROI_foreground_pixels','ROI_foreground_pixel2','final_ROI_area_pixel2','clean_ROI_area_pixel2') if k in areas),None)
            if pxkey is None:raise ValueError('Saved block ROI area field is missing: '+str(maskstats))
            values=areas[pxkey].to_numpy(float)*(config.get('acquisition',{}).get('pixel_size_nm',117)/1000)**2
        else:values=areas[key].to_numpy(float)
        weights=areas['source_frames'].to_numpy(float) if 'source_frames' in areas else (areas.original_frame_stop_exclusive-areas.original_frame_start).to_numpy(float) if {'original_frame_start','original_frame_stop_exclusive'}<=set(areas) else None
        if weights is None:raise ValueError('Block durations are missing; summarize does not infer time weights')
        record['first_mask_ROI_area_um2']=float(values[0]);record['mean_ROI_area_um2']=float(values@weights/weights.sum())
    else:raise FileNotFoundError('Saved cell metrics missing: '+sample.id+'; summarize cannot run upstream science')
    for key in COUNT_FIELDS:
        value=record[key]
        if int(value)!=value or value<0:raise ValueError('Invalid localization count '+sample.id+' '+key)
    if record['On']+record['Not_on']+record['unknown']!=record['selected'] or record['selected']+record['QC_excluded']!=record['ROI_candidates']:raise ValueError('Localization count conservation failed '+sample.id)
    record['total_localizations']=record['selected'];record['on_denominator']=record['On']+record['Not_on'];record['valid_metrics']={key:record.get(key) is not None and np.isfinite(record[key]) for key in METRICS}
    directory=decomp.file('directory') if decomp else Path(old)/'data/tardis' if old else None
    if directory is None:raise FileNotFoundError('Saved Intra artifact missing: '+sample.id)
    metapath=decomp.files.get('metadata') if decomp else None
    metadata=load_json(metapath or directory/'metadata.json');rules=metadata.get('config_rules',decomp.metadata.get('config_rules',metadata) if decomp else metadata)
    curves={};edges=None;reasons=[]
    summary=pd.read_csv(directory/'per_scope_summary.csv')
    for group in GROUPS:
        d=pd.read_csv(directory/'distribution'/group/'full.csv',float_precision='round_trip');left=d.distance_left_um.to_numpy(float);right=d.distance_right_um.to_numpy(float)
        if edges is None:edges=np.r_[left,right[-1]]
        elif not np.allclose(edges,np.r_[left,right[-1]],rtol=0,atol=1e-12):raise ValueError('Groups use different bins '+sample.id)
        curve=d.intra_density_um_inv.to_numpy(float);curves[group]=curve;area=float(np.sum(curve*(right-left)))
        row=summary[(summary.group==group)&(summary.scope=='full')]
        if len(row)!=1:reasons.append(group+':missing_validity_record')
        elif str(row.iloc[0].get('valid',False)).lower()!='true':reasons.append(group+':original_Intra_invalid')
        if not np.isfinite(curve).all():reasons.append(group+':nonfinite_curve')
        elif abs(area)<=.01:reasons.append(group+':near_zero_signed_area')
        elif float(np.max(curve))<=0:reasons.append(group+':no_positive_peak')
    method=dict(curve_rules={k:rules.get(k) for k in ('tau_frames','distance_bins','distance_max_um','background_fit_um','minimum_total_tail_pairs','minimum_background_tail_ordered_pairs','signed_area_min_abs','group_rule','normalization_rule','beta_rule','shape_rule')},method_versions={k:[v.method_id,v.method_version,v.parameters] for k,v in artifacts.items() if k in ('fit','broad','support_counts','mask_threshold','mask_cleanup','assignment','distances','decomposition')})
    # Source-dependent artifact IDs never define biological-method compatibility.
    if not artifacts:method['legacy_scientific_definition']=metadata.get('scientific_fingerprint',config.get('legacy_scientific_fingerprint'))
    return record,dict(sample_id=sample.id,status='complete',eligible=not reasons,reasons=reasons,curves=curves,edges=edges,method_definition=method,result_digest=digest({g:np.nan_to_num(curves[g],nan=0).tolist() for g in GROUPS}))


def _metric_nodes(records,root):
    brgroups=defaultdict(list);cgroups=defaultdict(list);nodes=[]
    for record in records:brgroups[(record['genotype'],record['BR'])].append(record);cgroups[record['genotype']].append(record)
    brstats={};conditionstats={}
    for key,rows in brgroups.items():brstats[key]={metric:_stats([r[metric] for r in rows],nbr=1) for metric in METRICS}
    for condition,rows in cgroups.items():
        children=[value for (c,_),value in brstats.items() if c==condition];conditionstats[condition]={metric:_stats([child[metric]['mean'] for child in children],ncells=sum(child[metric]['n_cells_valid'] for child in children),nbr=sum(child[metric]['n']>0 for child in children)) for metric in METRICS}
    for record in records:
        nodes.append(dict(level='cell',key=record['sample_id'],directory=str(root/record['source_directory_relative']),records=[record],BR_records=[],summary_records=[dict(condition=record['genotype'],metric=metric,**_stats([record.get(metric)],nbr=1)) for metric in METRICS]))
    for key,rows in brgroups.items():nodes.append(dict(level='BR',key='/'.join(key),directory=str((root/rows[0]['source_directory_relative']).parent),records=rows,BR_records=[dict(condition=key[0],BR=key[1],**brstats[key][metric],metric=metric) for metric in METRICS],summary_records=[dict(condition=key[0],BR=key[1],**brstats[key][metric],metric=metric) for metric in METRICS]))
    for condition,rows in cgroups.items():nodes.append(dict(level='condition',key=condition,directory=str(root/condition),records=rows,BR_records=[dict(condition=c,BR=br,metric=metric,**values[metric]) for (c,br),values in brstats.items() if c==condition for metric in METRICS],summary_records=[dict(condition=condition,metric=metric,**conditionstats[condition][metric]) for metric in METRICS]))
    nodes.append(dict(level='comparison',key='WT_TDK_MDK',directory=str(root),records=records,BR_records=[dict(condition=c,BR=br,metric=metric,**values[metric]) for (c,br),values in brstats.items() for metric in METRICS],summary_records=[dict(condition=c,metric=metric,**values[metric]) for c,values in conditionstats.items() for metric in METRICS]))
    return nodes


def _summary_input_signature(config,entries,method_config,root):
    """Immutable stage IDs suffice when both consumed scientific caches exist.

    Historical direct-file inputs without stage identities deliberately miss
    this fast path; they still benefit from unchanged per-node output reuse.
    """
    signatures=[]
    for sample,artifacts in entries:
        if not {'statistics','decomposition'}<=set(artifacts):return None
        identity={key:sample.metadata.get(key) for key in ('genotype','condition','BR','replica','cell','molecule','acquisition_date')}
        identity.update(sample_id=sample.id,source_directory_relative=sample.directory.relative_to(root).as_posix())
        signatures.append(dict(identity=identity,scientific_artifacts={key:value.id for key,value in artifacts.items() if key in ('fit','assignment','distances','decomposition','statistics','mask_roi')}))
    expected=[{key:row.get(key) for key in ('sample_id','genotype','condition','BR','replica','cell','molecule','source_directory_relative')} for row in _expected_inventory(config,[],root)]
    return digest(_plain(dict(summary=method_config,scientific_definitions={key:value for key,value in config.get('methods',{}).items() if key not in ('presentation','c1_mean')},inputs=sorted(signatures,key=lambda x:x['identity']['sample_id']),expected=sorted(expected,key=lambda x:x.get('sample_id','')))))


def _summary_numeric_files_exist(manifest,root):
    return bool(manifest.get('nodes')) and all(all((_portable(root,node['data'])/name).is_file() for name in node.get('numeric_files',{})) and node.get('numeric_files') for node in manifest.get('nodes',[]))


def summarize(config,samples,artifacts_by_sample=None,output_dir=None):
    """Compute hierarchy statistics explicitly; does not run any cell pipeline."""
    method_config=_summary_parameters(config)
    science_paths();root=Path(config['data_root']).resolve();control=Path(config.get('control_root',root/'_batch_pipeline'));records=[];cells={};inventory=[];sources=[]
    entries=list(_samples(samples,artifacts_by_sample));manifestdir=Path(output_dir) if output_dir else control/'modular_summaries';manifestpath=manifestdir/'summary_manifest.json';previous=load_json(manifestpath) if manifestpath.is_file() else {}
    input_signature=_summary_input_signature(config,entries,method_config,root)
    if input_signature is not None and previous.get('input_signature')==input_signature and _summary_numeric_files_exist(previous,root):
        return dict(manifest=str(manifestpath),summary_id=previous['summary_id'],nodes=previous['nodes'],reused=True,scientific_statistics_computed=False,upstream_stages_run=[],new_cells_started=0)
    for sample,artifacts in entries:
        record,cell=_load_cell(sample,artifacts,config);records.append(record);cells[sample.id]=cell;inventory.append(record);sources.append(dict(sample_id=sample.id,artifacts={k:v.id for k,v in artifacts.items() if k in ('statistics','decomposition','assignment','mask_roi')},scalar_digest=digest({key:record.get(key) for key in (*COUNT_FIELDS,*METRICS)}),curve_digest=cell['result_digest'],eligible=cell['eligible'],reasons=cell['reasons']))
    records.sort(key=lambda r:r['sample_id']);inventory.sort(key=lambda r:r['sample_id']);sources.sort(key=lambda r:r['sample_id'])
    inventory=_expected_inventory(config,inventory,root)
    methods={digest(cell['method_definition']) for cell in cells.values()}
    if len(methods)>1:
        # Legacy and newly computed adapter representations can differ. Explicit
        # provenance must resolve this; silently choosing one would mix methods.
        raise ValueError('Incompatible scientific definitions in hierarchy; separate runs/configurations are required')
    if not inventory:raise ValueError('No expected or completed cells available for summarize')
    import intra_summary_core as core
    edges=next(iter(cells.values()))['edges'] if cells else np.linspace(0,3,301)
    if edges.shape!=(301,) or not np.allclose(edges,np.linspace(0,3,301),rtol=0,atol=1e-12):raise ValueError('Current hierarchy summary method supports the frozen 0–3 µm / 300-bin definition only')
    nodeobjects=core.summarize_cells(inventory,cells,dict(data_root=str(root)))
    for node in nodeobjects:node['molecule']=config.get('molecule',records[0].get('molecule','') if records else '')
    summaryid=digest(dict(method_id='hierarchy_summary',method_version='1',parameters=method_config,scientific_definitions=sorted(methods)))[:16]
    manifestdir.mkdir(parents=True,exist_ok=True)
    metricnodes=_metric_nodes(records,root);jobs=[];nodes=[]
    from reader_table_data import build_reader_tables
    for metricnode in metricnodes:
        directory=Path(metricnode['directory']);data=directory/'analysis_results'/('modular_summary_'+summaryid);data.mkdir(parents=True,exist_ok=True)
        matching=next((n for n in nodeobjects if n['key']==metricnode['key'] and n['level']==metricnode['level']),None)
        numeric_records=[{key:record.get(key) for key in ('sample_id','genotype','BR','cell','molecule','acquisition_date',*COUNT_FIELDS,*METRICS)} for record in metricnode['records']]
        numeric_signature=digest(_plain(dict(records=numeric_records,summary=metricnode['summary_records'],BR=metricnode['BR_records'],intra={key:matching.get(key) for key in ('groups','members','exclusions','progress')} if matching else None)))
        prior=next((n for n in previous.get('nodes',[]) if n['level']==metricnode['level'] and n['key']==metricnode['key']),None)
        if previous.get('summary_id')==summaryid and prior and prior.get('numeric_signature')==numeric_signature and prior.get('numeric_files') and all((data/name).is_file() for name in prior.get('numeric_files',{})):
            nodes.append(prior);continue
        # Full-precision reader data; operational provenance remains separate.
        _save_csv(pd.DataFrame(metricnode['records']).drop(columns=['valid_metrics'],errors='ignore'),data/'cell_metrics.csv')
        _save_csv(pd.DataFrame(metricnode['summary_records']),data/'summary_statistics.csv');_save_csv(pd.DataFrame(metricnode['BR_records']),data/'BR_statistics.csv')
        from localization_count_data import count_csv_specs
        noderecord=dict(level=metricnode['level'],key=metricnode['key'],data=str(data),molecule=config.get('molecule'))
        reader=dict(key=metricnode['key'],level=metricnode['level'],data_dir=str(data),output_path=str(directory/'Analysis_Figures_Cell_Metrics'/'Cell_Metrics.xlsx'),sheets=build_reader_tables(noderecord,root))
        for spec in count_csv_specs(reader):_save_csv(pd.DataFrame(spec['rows'],columns=spec['headers']),data/spec['filename'])
        dump(data/'reader_workbook_job.json',reader);dump(data/'metric_node.json',metricnode)
        matching=next((n for n in nodeobjects if n['key']==metricnode['key'] and n['level']==metricnode['level']),None)
        if matching:
            dump(data/'intra_node.json',matching)
            rows=[]
            for group in GROUPS:
                for view in VIEWS:
                    part=matching['groups'][group][view]
                    series=dict(part['series']);
                    if part.get('mean') is not None:series['__mean__']=part['mean']
                    for sid,curve in series.items():
                        for i,value in enumerate(curve):rows.append(dict(group=group,normalization=view,series_id=sid,distance_mid_um=matching['x'][i],value=value,SD=part.get('sd')[i] if sid=='__mean__' and part.get('sd') is not None else None,SEM=part.get('sem')[i] if sid=='__mean__' and part.get('sem') is not None else None,n=part['n']))
            _save_csv(pd.DataFrame(rows),data/'intra_curves_long.csv.gz',compression='gzip')
            dump(data/'cohort_membership.json',dict(members=matching['members'],exclusions=matching['exclusions'],progress=matching['progress']))
        numeric_records=[{key:record.get(key) for key in ('sample_id','genotype','BR','cell','molecule','acquisition_date',*COUNT_FIELDS,*METRICS)} for record in metricnode['records']]
        numeric_signature=digest(_plain(dict(records=numeric_records,summary=metricnode['summary_records'],BR=metricnode['BR_records'],intra={key:matching.get(key) for key in ('groups','members','exclusions','progress')} if matching else None)))
        numeric_files=[data/'metric_node.json',data/'reader_workbook_job.json']+sorted(data.glob('*.csv'))+sorted(data.glob('*.csv.gz'))+([data/'intra_node.json',data/'cohort_membership.json'] if matching else [])
        nodes.append(dict(level=metricnode['level'],key=metricnode['key'],directory=directory.relative_to(root).as_posix(),data=data.relative_to(root).as_posix(),intra=bool(matching),workbook_job=(data/'reader_workbook_job.json').relative_to(root).as_posix(),numeric_signature=numeric_signature,numeric_files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in numeric_files}))
    manifest=dict(schema_version=1,summary_id=summaryid,input_signature=input_signature,nodes=nodes,method=method_config,source_scientific_definitions=sorted(methods),sources=sources,scientific_statistics_computed=True,upstream_stages_run=[],new_cells_started=0)
    dump(manifestdir/'summary_manifest.json',manifest)
    return dict(manifest=str(manifestdir/'summary_manifest.json'),summary_id=summaryid,nodes=nodes,reused=False,scientific_statistics_computed=True,upstream_stages_run=[],new_cells_started=0)


def _metric_plot(node,figures):
    import matplotlib.pyplot as plt
    records=node['records'];fig,axes=plt.subplots(1,3,figsize=(15,5));colors={'WT':'#2166AC','TDK':'#D95F02','MDK':'#1B9E77'}
    conditions=list(dict.fromkeys(r['genotype'] for r in records))
    for ax,metric in zip(axes,METRICS):
        for i,condition in enumerate(conditions):
            rows=[r for r in records if r['genotype']==condition];x=np.full(len(rows),i);ax.scatter(x,[r.get(metric,np.nan) for r in rows],color=colors.get(condition,'gray'),s=12,alpha=.6)
            saved=[r for r in node['summary_records'] if r['condition']==condition and r['metric']==metric]
            if len(saved)==1 and saved[0]['mean'] is not None:ax.errorbar(i,saved[0]['mean'],yerr=saved[0]['sd'],fmt='s',color='black',capsize=3)
        ax.set(xticks=range(len(conditions)),xticklabels=conditions,title=metric,ylabel='%' if metric=='on_percentage' else 'ROI mask area (µm²)');ax.grid(alpha=.2)
    fig.suptitle(node['key']+' | saved cell / replica statistics');fig.tight_layout();temporary=figures/('.Cell_metrics.'+uuid.uuid4().hex+'.png');fig.savefig(temporary,dpi=180);plt.close(fig);atomic_replace(temporary,figures/'Cell_metrics.png')


def _overlays(node,figures):
    import matplotlib.pyplot as plt
    colors={'On':'#00B894','Not-on':'#A45BDB'};comparison=node['level']=='comparison'
    for view in VIEWS:
        panels=list(node['groups']['On'][view]['series']) if comparison else [None];panels=panels or [None]
        traces=[]
        for panel in panels:
            for group in GROUPS:
                p=node['groups'][group][view];value=p['series'].get(panel) if comparison else p['mean'];sd=p.get('series_sd',{}).get(panel) if comparison else p['sd']
                if value is not None:
                    a=np.asarray(value);spread=np.asarray(sd) if sd is not None else 0;traces.extend([a-spread,a+spread])
        lo=min([0]+[float(np.min(a)) for a in traces]);hi=max([0]+[float(np.max(a)) for a in traces]);pad=.07*(hi-lo) if hi>lo else .05;limits=[lo-pad,hi+pad]
        for range_name,xmax in [('full',3),('zoom',.5)]:
            fig,axes=plt.subplots(1,len(panels),figsize=(6*len(panels),5),squeeze=False,sharey=True)
            for ax,panel in zip(axes[0],panels):
                for group in GROUPS:
                    p=node['groups'][group][view];value=p['series'].get(panel) if comparison else p['mean'];sd=p.get('series_sd',{}).get(panel) if comparison else p['sd']
                    if value is None:continue
                    value=np.asarray(value);ax.plot(node['x'],value,color=colors[group],label=group)
                    if sd is not None:sd=np.asarray(sd);ax.fill_between(node['x'],value-sd,value+sd,color=colors[group],alpha=.15)
                ax.set(title=panel or node['key'],xlim=(0,xmax),ylim=limits,xlabel='Distance (µm)',ylabel='Dimensionless' if view=='peak' else 'Density (µm⁻¹)');ax.axhline(0,color='gray',lw=.6);ax.legend();ax.grid(alpha=.2)
            fig.suptitle(node['progress']['status']+' | On / Not-on | '+view);fig.tight_layout();target=figures/f'Intra_On_NotOn_{view}_{range_name}.png';temporary=figures/('.'+uuid.uuid4().hex+'_'+target.name);fig.savefig(temporary,dpi=180);plt.close(fig);atomic_replace(temporary,target)


def refresh(config,manifest=None):
    """Pure publisher. Even absent products result in zero scientific calls."""
    science_paths();root=Path(config['data_root']).resolve();path=Path(manifest) if manifest else Path(config.get('control_root',root/'_batch_pipeline'))/'modular_summaries/summary_manifest.json'
    if not path.is_file():raise FileNotFoundError('Saved summary manifest is missing; run summarize explicitly. refresh performs no science')
    saved=load_json(path);reports=[];missing=[]
    from export_reader_tables import write_workbook
    from intra_summary_plots import render_node
    import matplotlib
    matplotlib.use('Agg')
    for node in saved['nodes']:
        data=_portable(root,node['data']);jobpath=_portable(root,node['workbook_job']);required=[data/'metric_node.json',jobpath]+([data/'intra_node.json'] if node['intra'] else [])
        absent=[str(p) for p in required if not p.is_file()]
        if absent:missing.extend(absent);continue
        for filename,expected_sha in node.get('numeric_files',{}).items():
            if hashlib.sha256((data/filename).read_bytes()).hexdigest()!=expected_sha:raise ValueError('Saved summary product differs from committed manifest: '+str(data/filename))
        directory=_portable(root,node['directory']);figures=directory/'Analysis_Figures_Cell_Metrics';figures.mkdir(exist_ok=True)
        renderer_sha=hashlib.sha256(Path(__file__).read_bytes()+(COMPAT/'intra_summary_plots.py').read_bytes()+(COMPAT/'export_reader_tables.py').read_bytes()).hexdigest()
        signature=digest(dict(numeric_signature=node.get('numeric_signature'),style=config.get('style',{}),renderer_audit_sha256=renderer_sha))
        expected=[figures/'Cell_metrics.png',figures/'Cell_Metrics.xlsx',figures/'INDEX.html']
        if node['intra']:
            expected += [directory/'Analysis_Figures_Intra'/f'Intra_{view}_{r}.png' for view in VIEWS for r in ('full','zoom')]
            expected += [directory/'Analysis_Figures_Intra/On_NotOn_Overlay'/f'Intra_On_NotOn_{view}_{r}.png' for view in VIEWS for r in ('full','zoom')]
            expected += [directory/'Analysis_Figures_Intra/INDEX.html',directory/'Analysis_Figures_Intra/On_NotOn_Overlay/INDEX.html']
        publication=data/'publication_node.json';prior=load_json(publication) if publication.is_file() else {}
        if prior.get('signature')==signature and all(p.is_file() and p.stat().st_size>0 for p in expected):
            reports.append(dict(key=node['key'],status='unchanged',workbook=str(figures/'Cell_Metrics.xlsx'),images=sum(p.suffix=='.png' for p in expected)));continue
        metricnode=load_json(data/'metric_node.json')
        _metric_plot(metricnode,figures)
        job=load_json(jobpath);template=COMPAT/'reader_table_templates'/('cell.xlsx' if node['level']=='cell' else 'BR.xlsx' if node['level']=='BR' else 'summary.xlsx');write_workbook(template,job,figures/'Cell_Metrics.xlsx')
        images=[figures/'Cell_metrics.png']
        if node['intra']:
            intra=load_json(data/'intra_node.json');intra['directory']=directory;intra['x']=np.asarray(intra['x']);intra['edges']=np.asarray(intra['edges']);intrafig=directory/'Analysis_Figures_Intra';intrafig.mkdir(exist_ok=True);render_node(intra,intrafig);overlay=intrafig/'On_NotOn_Overlay';overlay.mkdir(exist_ok=True);_overlays(intra,overlay);images+=list(intrafig.glob('*.png'))+list(overlay.glob('*.png'))
        for directory in {p.parent for p in images}:
            entries=''.join('<li><a href="'+html.escape(p.name)+'">'+html.escape(p.name)+'</a><br><img src="'+html.escape(p.name)+'" width="700"></li>' for p in images if p.parent==directory)
            (directory/'INDEX.html').write_text('<html><body><h1>'+html.escape(node['key'])+'</h1><p>Saved numeric summary only; no scientific recomputation.</p><ul>'+entries+'</ul></body></html>',encoding='utf-8')
        dump(publication,dict(signature=signature,scientific_stage_calls=0))
        reports.append(dict(key=node['key'],status='updated',workbook=str(figures/'Cell_Metrics.xlsx'),images=len(images)))
    report=dict(summary_id=saved['summary_id'],nodes=reports,missing=missing,scientific_stage_calls=0,new_cells_started=0)
    dump(path.parent/'publication_report.json',report)
    if missing:raise FileNotFoundError('Saved summary products missing; no science was run: '+', '.join(missing))
    return report
