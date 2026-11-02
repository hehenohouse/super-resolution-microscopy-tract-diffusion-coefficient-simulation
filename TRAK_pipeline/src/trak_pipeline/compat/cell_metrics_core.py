"""Cached-only cell metrics and independent cell/BR/condition summaries.

No image, mask-array, localization-table or Intra-curve data are read here.
"""
from pathlib import Path
from collections import defaultdict
import csv, hashlib, importlib.util, json, math, statistics

DEFAULT_FINGERPRINT='58bef6b94294f648f15e8637a744e71c6dd59a3fa0fd4e40c483a98d204cb7a3'
METRICS=('on_percentage','first_mask_ROI_area_um2','mean_ROI_area_um2')
UNITS={'on_percentage':'percent','first_mask_ROI_area_um2':'um2','mean_ROI_area_um2':'um2'}
RECIPE='native_broad_support50_ge10_postmin32_FG8_holes_lt64_BG4'
COUNT_FIELDS=('On','Not_on','unknown','QC_excluded','selected','ROI_candidates')


def _json(path):
    with Path(path).open(encoding='utf-8') as stream:return json.load(stream)


def _true(value):return value is True or str(value).lower() in ('true','1')


def _reference(root,path):
    path=Path(path);stat=path.stat();h=hashlib.sha256()
    with path.open('rb') as stream:
        if stat.st_size<=2*1024*1024:h.update(stream.read());kind='complete_small_file'
        else:
            h.update(stream.read(65536));stream.seek(max(0,stat.st_size-65536));h.update(stream.read(65536));kind='first_last_64KiB'
    return dict(relative_path=path.resolve().relative_to(root).as_posix(),bytes=stat.st_size,sha256=h.hexdigest(),fingerprint_kind=kind)


def _unique_refs(refs):
    result={}
    for ref in refs:result[(ref['relative_path'],ref['sha256'])]=ref
    return [result[key] for key in sorted(result)]


def _integer(value,name):
    if isinstance(value,bool):raise ValueError(name+': boolean is not an integer count')
    number=float(value)
    if not math.isfinite(number) or number!=int(number) or number<0:raise ValueError(name+': expected nonnegative integer')
    return int(number)


def _blank(row,status,reason=None):
    record={key:row.get(key) for key in ('sample_id','genotype','BR','cell','molecule','acquisition_date')}
    record.update(source_run_id=None,source_output_relative=None,status=status,source_refs=[],equivalent_outputs_relative=[])
    record.update({key:None for key in COUNT_FIELDS});record['on_denominator']=None
    record.update({key:None for key in METRICS})
    record['valid_metrics']={key:False for key in METRICS}
    record['reasons']={key:([reason] if reason else []) for key in METRICS}
    return record


def _issue(record,metrics,reason):
    for metric in metrics:
        record['valid_metrics'][metric]=False;record[metric]=None
        if reason not in record['reasons'][metric]:record['reasons'][metric].append(reason)


def _stats(values,metric,n_cells_valid=None,n_BRs_valid=0,unit_type="cell"):
    values=[float(value) for value in values if value is not None and math.isfinite(float(value))]
    n=len(values);mean=statistics.fmean(values) if n else None
    sd=statistics.stdev(values) if n>1 else None
    return dict(mean=mean,sd=sd,sem=(sd/math.sqrt(n) if sd is not None else None),n=n,unit_type=unit_type,units=UNITS[metric],n_cells_valid=n if n_cells_valid is None else n_cells_valid,n_BRs_valid=n_BRs_valid)


def _validate_mask_metadata(meta,sid):
    def get(*keys):
        for key in keys:
            if key in meta:return meta[key]
        raise ValueError('mask metadata field missing: '+str(keys))
    if meta.get('sample_id',sid)!=sid:raise ValueError('mask sample ID mismatch')
    if list(get('shape'))!=[50,684,428] or get('axes')!='TYX':raise ValueError('mask TYX shape mismatch')
    if int(get('original_frames','source_frame_count'))!=2500:raise ValueError('mask original frame count mismatch')
    if float(get('pixel_size_nm'))!=117:raise ValueError('mask pixel size mismatch')
    if float(get('original_frame_interval_s','source_frame_interval_s','frame_interval_s'))!=.02:raise ValueError('mask frame interval mismatch')
    if int(get('mask_block_frames','support_total_frames'))!=50:raise ValueError('mask block length mismatch')
    if int(get('foreground_support_minimum','support_minimum_count'))!=10:raise ValueError('mask support threshold mismatch')
    if int(get('minimum_foreground_area_pixel2'))!=32:raise ValueError('mask foreground threshold mismatch')
    if int(get('hole_area_exclusive_max_pixel2'))!=64:raise ValueError('mask hole threshold mismatch')
    if int(get('foreground_connectivity'))!=8 or int(get('hole_background_connectivity','background_connectivity'))!=4:raise ValueError('mask connectivity mismatch')
    if meta.get('recipe',RECIPE)!=RECIPE:raise ValueError('mask recipe mismatch')
    if _true(meta.get('per_frame_extra_min32',meta.get('per_frame_min32',False))):raise ValueError('unexpected pre-support min32')


def _load_output(root,entry,row,portable_path,expected_fingerprint):
    record=_blank(row,'complete');refs=[];sid=row['sample_id']
    try:
        output=portable_path(root,entry.get('result_directory_relative',entry.get('output','')))
        cell=portable_path(root,row['source_directory_relative'])
        if not output.is_relative_to(cell):raise ValueError('completion belongs to another cell')
        record['source_output_relative']=output.relative_to(root).as_posix()
        marker_path=output/'COMPLETED.json';marker=_json(marker_path);refs.append(_reference(root,marker_path))
        if not _true(marker.get('complete')) or marker.get('sample_id')!=sid:raise ValueError('committed completion mismatch')
        if marker.get('scientific_fingerprint')!=expected_fingerprint or entry.get('scientific_fingerprint',expected_fingerprint)!=expected_fingerprint:raise ValueError('scientific fingerprint mismatch')
        index_path=output/'RESULT_INDEX.json';index=_json(index_path);refs.append(_reference(root,index_path))
        if index.get('sample_id')!=sid or index.get('scientific_fingerprint')!=expected_fingerprint:raise ValueError('result index identity/fingerprint mismatch')
        # Resolve import wrappers without copying numerical data.
        seen={output.resolve()}
        while index.get('source_result_directory') and not index.get('data') and not index.get('source_directories'):
            output=portable_path(root,index['source_result_directory'])
            if output in seen or not output.is_relative_to(cell):raise ValueError('invalid source-result wrapper')
            seen.add(output);index_path=output/'RESULT_INDEX.json';index=_json(index_path);refs.append(_reference(root,index_path))
            if index.get('sample_id')!=sid or index.get('scientific_fingerprint')!=expected_fingerprint:raise ValueError('canonical index identity/fingerprint mismatch')
        record['source_run_id']=index.get('run_id')
        record['registration_status']=index.get('registration_status','exploratory_identity_registration_unverified')
        record['calibration_status']=index.get('calibration_status','manufacturer_nominal')
    except (OSError,ValueError,KeyError,TypeError) as error:
        record['status']='failed';_issue(record,METRICS,'invalid_completion: '+str(error));record['source_refs']=_unique_refs(refs);return record
    try:
        classification_path=output/'classification_summary.json'
        counts=index.get('counts',{})
        if classification_path.is_file():
            classification=_json(classification_path);refs.append(_reference(root,classification_path))
            if classification.get('sample_id',sid)!=sid:raise ValueError('classification sample ID mismatch')
            if int(classification.get('original_frames',2500))!=2500 or float(classification.get('pixel_size_nm',117))!=117 or float(classification.get('pointwise_k',1))!=1:raise ValueError('classification frame/scale/k mismatch')
            for key in COUNT_FIELDS:
                if key in counts and _integer(counts[key],key)!=_integer(classification[key],key):raise ValueError('classification/index count mismatch: '+key)
            counts=classification
        parsed={key:_integer(counts[key],key) for key in COUNT_FIELDS}
        if parsed['On']+parsed['Not_on']+parsed['unknown']!=parsed['selected']:raise ValueError('spatial count conservation failed')
        if parsed['selected']+parsed['QC_excluded']!=parsed['ROI_candidates']:raise ValueError('candidate count conservation failed')
        record.update(parsed);denominator=parsed['On']+parsed['Not_on'];record['on_denominator']=denominator
        if denominator:
            record['on_percentage']=100.0*parsed['On']/denominator;record['valid_metrics']['on_percentage']=True
        else:_issue(record,['on_percentage'],'zero_On_plus_Not_on_denominator')
    except (OSError,ValueError,KeyError,TypeError) as error:_issue(record,['on_percentage'],'invalid_counts: '+str(error))
    area_metrics=METRICS[1:]
    try:
        source_dirs=index.get('source_directories',{})
        masks=portable_path(root,index.get('masks') or source_dirs.get('mask_directory') or str(output/'masks'))
        if not masks.is_relative_to(cell):raise ValueError('mask directory belongs to another cell')
        metadata_path=masks/'metadata.json'
        if not metadata_path.is_file():metadata_path=masks/'parameters.json'
        meta=_json(metadata_path);refs.append(_reference(root,metadata_path));_validate_mask_metadata(meta,sid)
        stats_path=masks/'per_block_statistics.csv';refs.append(_reference(root,stats_path))
        with stats_path.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        if not rows:raise ValueError('mask block area table is empty')
        areas=[];weights=[]
        pixel_area=(117.0/1000.0)**2
        for block,rowdata in enumerate(rows):
            start=_integer(rowdata['original_frame_start'],'frame_start')
            stop=_integer(rowdata['original_frame_stop_exclusive'],'frame_stop') if 'original_frame_stop_exclusive' in rowdata else _integer(rowdata['original_frame_end_inclusive'],'frame_end')+1
            if _integer(rowdata['block_index'],'block_index')!=block or start!=block*50 or stop!=(block+1)*50:raise ValueError('mask block mapping mismatch')
            area=_integer(rowdata.get('mask_ROI_area_px2',rowdata.get('ROI_pixel2')),'ROI area')
            full=_integer(rowdata.get('mask_full_area_px2',rowdata.get('full_pixel2')),'full area')
            if area>full or full>684*428:raise ValueError('ROI/full mask area bounds failed')
            areas.append(area);weights.append(stop-start)
            if block==0:
                record['first_mask_ROI_area_um2']=area*pixel_area
                record['valid_metrics']['first_mask_ROI_area_um2']=True
        if len(rows)!=50:raise ValueError('expected 50 mask blocks')
        record['mean_ROI_area_um2']=sum(a*w for a,w in zip(areas,weights))/sum(weights)*pixel_area
        for key in area_metrics:record['valid_metrics'][key]=True
        record['mask_source_relative']=masks.relative_to(root).as_posix()
        record['first_mask_original_frame_start']=0;record['first_mask_original_frame_stop_exclusive']=50
    except (OSError,ValueError,KeyError,TypeError) as error:
        unavailable=[metric for metric in area_metrics if not record['valid_metrics'][metric]]
        _issue(record,unavailable,'invalid_mask_area_source: '+str(error))
    record['source_refs']=_unique_refs(refs)
    return record


def _scalar_digest(record):
    payload={key:record.get(key) for key in (*COUNT_FIELDS,'on_denominator',*METRICS,'valid_metrics','reasons','status')}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _progress(records,expected_BRs=0):
    completed=sum(row['status']=='complete' for row in records);failed=sum(row['status'] in ('failed','conflict') for row in records)
    included=sum(row['status']=='complete' and any(row['valid_metrics'].values()) for row in records)
    fully_valid=sum(row['status']=='complete' and all(row['valid_metrics'].values()) for row in records)
    status='COMPLETE' if completed==len(records) and fully_valid==len(records) else ('COMPLETE_WITH_FAILURES_OR_EXCLUSIONS' if completed+failed==len(records) else 'PROGRESS')
    return dict(completed=completed,expected=len(records),included=included,failed=failed,status=status,valid_BRs=0,expected_BRs=expected_BRs)


def _node(level,key,directory,records,condition=None,BR=None,expected_BRs=0):
    return dict(level=level,key=key,directory=Path(directory),condition=condition,BR=BR,cell_records=records,metric_summaries={},BR_records=[],condition_records=[],source_refs=_unique_refs([ref for record in records for ref in record.get('source_refs',[])]),progress=_progress(records,expected_BRs),blocked=any(record['status']=='conflict' for record in records))


def build_metrics(control,release,config):
    """Read committed scalar caches and return inventory-derived hierarchy nodes, without writes."""
    control=Path(control).resolve();release=Path(release).resolve();root=control.parent
    expected_fingerprint=config.get('scientific_fingerprint')
    if config.get('molecule') not in ('TRAK1','TRAK2'):raise ValueError('Unsupported cell metric molecule')
    if not isinstance(expected_fingerprint,str) or len(expected_fingerprint)!=64 or any(c not in '0123456789abcdef' for c in expected_fingerprint):raise ValueError('Invalid scientific fingerprint')
    required=dict(schema_version=1,data_root='@portable_data_root',molecule=config['molecule'],condition_order=['WT','TDK','MDK'],replicas=['BR1','BR2','BR3'],frames=2500,pixel_size_nm=117,frame_interval_s=.02,pointwise_k=1,on_denominator='On+Not_on',cell_weighting='equal',BR_weighting='equal',SD_ddof=1,scientific_fingerprint=expected_fingerprint)
    for key,value in required.items():
        if config.get(key)!=value:raise ValueError('Unsupported cell metric configuration: '+key)
    spec=importlib.util.spec_from_file_location('_cell_metrics_shared_intra_inventory',release/'intra_summary_core.py');shared=importlib.util.module_from_spec(spec);spec.loader.exec_module(shared)
    inventory=shared.build_inventory(control,release);portable_path=shared.portable_path
    if any(row.get('molecule')!=config['molecule'] for row in inventory):raise ValueError('Inventory molecule differs from the metrics configuration')
    expected={row['sample_id']:row for row in inventory};candidates=defaultdict(list);observed=defaultdict(list);errors=[];registry_refs=[]
    state_paths=[control/'state/active_batch.json',control/'windows/state/active_batch.json']
    winstate=control/'windows/state'
    if winstate.is_dir():state_paths.extend(sorted(winstate.glob('*/active_batch.json')))
    portable_paths=[control/'windows/portable_completion_registry.json',winstate/'completed_cells_portable.json']
    if winstate.is_dir():portable_paths.extend(sorted(winstate.glob('*/completed_cells_portable.json')))
    for path in state_paths+portable_paths:
        if not path.is_file():continue
        try:
            document=_json(path);registry_refs.append(_reference(root,path))
            for sid,item in document.get('cells',document.get('entries',{})).items():
                if sid not in expected:continue
                observed[sid].append(item.get('status','pending'))
                committed=item.get('status') in ('complete','completed','reused') or _true(item.get('complete'))
                if committed:
                    record=dict(item);record.setdefault('scientific_fingerprint',document.get('scientific_fingerprint',document.get('source_scientific_fingerprint',expected_fingerprint)));candidates[sid].append(record)
        except (OSError,ValueError,TypeError) as error:errors.append(dict(stage='registry',source=str(path),reason=str(error)))
    cells=[]
    for sid,row in expected.items():
        unique={};invalid=[]
        for item in candidates[sid]:
            try:
                path=portable_path(root,item.get('result_directory_relative',item.get('output','')))
                key=str(path).casefold()
                if key not in unique:unique[key]=_load_output(root,item,row,portable_path,expected_fingerprint)
            except (OSError,ValueError,KeyError,TypeError) as error:invalid.append(_blank(row,'failed','invalid_completion_path: '+str(error)))
        alternatives=list(unique.values())+invalid
        if not alternatives:
            failed=any(value in ('failed','error','invalid','invalid_input') for value in observed[sid]);cell=_blank(row,'failed' if failed else 'pending','upstream_cell_failed' if failed else 'not_completed')
        elif len({_scalar_digest(record) for record in alternatives})>1:
            cell=_blank(row,'conflict','conflicting_committed_scalar_results');cell['source_refs']=_unique_refs([ref for alternative in alternatives for ref in alternative['source_refs']]);cell['conflicting_outputs_relative']=[a.get('source_output_relative') for a in alternatives]
        else:
            cell=sorted(alternatives,key=lambda value:str(value.get('source_output_relative')))[0];cell['equivalent_outputs_relative']=sorted({a['source_output_relative'] for a in alternatives if a.get('source_output_relative')});cell['source_refs']=_unique_refs([ref for alternative in alternatives for ref in alternative['source_refs']])
        cells.append(cell)
    nodes=[];br_nodes=[];condition_nodes=[]
    for cell in cells:
        expected_row=expected[cell['sample_id']];node=_node('cell',cell['sample_id'],portable_path(root,expected_row['source_directory_relative']),[cell],cell['genotype'],cell['BR'])
        for metric in METRICS:node['metric_summaries'][metric]=_stats([cell[metric]] if cell['valid_metrics'][metric] else [],metric)
        nodes.append(node)
    for condition in config['condition_order']:
        condition_cells=[cell for cell in cells if cell['genotype']==condition]
        for BR in config['replicas']:
            records=[cell for cell in condition_cells if cell['BR']==BR]
            matching=[row for row in inventory if row['genotype']==condition and row['BR']==BR]
            if not matching:continue
            directory=portable_path(root,matching[0]['source_directory_relative']).parent
            node=_node('BR',condition+'/'+BR,directory,records,condition,BR,1)
            for metric in METRICS:
                values=[record[metric] for record in records if record['status']=='complete' and record['valid_metrics'][metric]] if not node['blocked'] else []
                node['metric_summaries'][metric]=_stats(values,metric,n_BRs_valid=int(bool(values)))
            node['progress']['valid_BRs']=int(any(value['n'] for value in node['metric_summaries'].values()));node['BR_records']=[dict(condition=condition,BR=BR,metrics=node['metric_summaries'])]
            br_nodes.append(node);nodes.append(node)
        children=[node for node in br_nodes if node['condition']==condition]
        node=_node('condition',condition,root/condition,condition_cells,condition,None,len(children))
        node['BR_records']=[dict(condition=condition,BR=child['BR'],metrics=child['metric_summaries']) for child in children]
        for metric in METRICS:
            valid=[child['metric_summaries'][metric] for child in children if child['metric_summaries'][metric]['n']>0] if not node['blocked'] else []
            node['metric_summaries'][metric]=_stats([value['mean'] for value in valid],metric,n_cells_valid=sum(value['n_cells_valid'] for value in valid),n_BRs_valid=len(valid),unit_type="BR")
        node['progress']['valid_BRs']=sum(any(value['n'] for value in child['metric_summaries'].values()) for child in children) if not node['blocked'] else 0
        node['condition_records']=[dict(condition=condition,metrics=node['metric_summaries'])];condition_nodes.append(node);nodes.append(node)
    top=_node('comparison','WT_TDK_MDK',root,cells,None,None,len(br_nodes));top['blocked']=False
    top['BR_records']=[dict(condition=node['condition'],BR=node['BR'],metrics=node['metric_summaries']) for node in br_nodes]
    top['condition_records']=[dict(condition=node['condition'],metrics=node['metric_summaries']) for node in condition_nodes]
    for metric in METRICS:
        valid=[node['metric_summaries'][metric] for node in condition_nodes if node['metric_summaries'][metric]['n']>0]
        top['metric_summaries'][metric]=dict(mean=None,sd=None,sem=None,n=len(valid),unit_type=None,units=UNITS[metric],n_cells_valid=sum(value['n_cells_valid'] for value in valid),n_BRs_valid=sum(value['n_BRs_valid'] for value in valid))
    top['progress']['valid_BRs']=sum(node['progress']['valid_BRs'] for node in condition_nodes);nodes.append(top)
    for node in nodes:node['molecule']=config['molecule']
    discovery=dict(registries=registry_refs,registered_samples=sum(bool(candidates[sid]) for sid in expected),deduplicated_cells=len(cells),status_counts={status:sum(cell['status']==status for cell in cells) for status in ('complete','pending','failed','conflict')},independent_metric_validity=True,no_Intra_eligibility_filter=True)
    return dict(nodes=nodes,cells=cells,discovery=discovery,errors=errors,expected_cells=len(inventory))
