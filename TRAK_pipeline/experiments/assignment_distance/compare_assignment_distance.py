#!/usr/bin/env python3
"""Compare saved k=1 assignment with exact fixed-radius mask proximity.

Only one explicitly selected cell is processed. All writes remain on My Book.
This tool does not detect, fit, segment, link trajectories, or run TARDIS.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter,defaultdict
from itertools import combinations
import argparse,csv,gzip,hashlib,importlib.metadata,json,math,os,re,shutil,subprocess,sys,time

HERE=Path(__file__).resolve().parent
VOLUME=Path('/Volumes/My Book')
CONTROL=HERE.parents[1]
SAMPLE_FRAMES=[0,49,200,625,1200,1250,1875,2000,2450,2499]
LABELS=['mito-compatible','non-mito-compatible','unknown']
DISPLAY={'mito-compatible':'On','non-mito-compatible':'Not_on','unknown':'unknown'}
BASELINE=dict(ROI_candidates=94204,selected=91860,QC_excluded=2344,On=32489,Not_on=59371,unknown=0)
RECIPE='native_broad_support50_ge10_postmin32_FG8_holes_lt64_BG4'

def utc():return datetime.now(timezone.utc).isoformat()
def stamp():return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
def guard(path):
    if not os.path.ismount(VOLUME):raise OSError('My Book is not mounted; local fallback is disabled')
    path=Path(path).absolute()
    if not path.resolve().is_relative_to(VOLUME.resolve()):raise ValueError('Write path is outside My Book')
    return path
def dump(path,value):
    path=guard(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def descriptor(path):
    path=Path(path);st=path.stat();d=dict(path=str(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns)
    if st.st_size<=2*1024*1024:d.update(sha256=hash_file(path),kind='complete_small_file')
    else:
        h=hashlib.sha256()
        with path.open('rb') as stream:
            h.update(stream.read(65536));stream.seek(max(0,st.st_size-65536));h.update(stream.read(65536))
        d.update(sampled_sha256=h.hexdigest(),kind='size_mtime_and_first_last_64KiB')
    return d
def flag(value):return str(value).strip().lower() in ('true','1')
def number(row,key):
    try:return float(row[key])
    except (ValueError,TypeError,KeyError):return float('nan')
def env_for(output):
    runtime=guard(output/'runtime');runtime.mkdir(parents=True,exist_ok=True)
    for key,name in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('MPLCONFIGDIR','matplotlib'),('XDG_CACHE_HOME','xdg'),('NUMBA_CACHE_DIR','numba')]:
        folder=runtime/name;folder.mkdir(exist_ok=True);os.environ[key]=str(folder)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',MPLBACKEND='Agg',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
    sys.dont_write_bytecode=True
def write_csv(path,rows,fields):
    path=guard(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
    with temp.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    os.replace(temp,path)
def event(config,text,**values):
    record=dict(utc=utc(),message=text,**values);line=json.dumps(record,allow_nan=False)
    print(line,flush=True)
    with (Path(config['output'])/'run.log').open('a',encoding='utf-8') as stream:stream.write(line+'\n')
def row_digest(h,row,fields):
    h.update(json.dumps([row[k] for k in fields],ensure_ascii=False,separators=(',',':')).encode());h.update(b'\n')
def selection(row):return flag(row.get('qc_unflagged_model_fit')) and flag(row.get('fitted_center_inside_ROI'))
def new_fields(radii):
    return [f'close_{r}px_{suffix}' for r in radii for suffix in ('label','unknown_reason','intersects_known_mask')]

def discover(cell,source_output=None):
    cell=guard(cell).resolve()
    candidates=[p.parent for p in (cell/'analysis_results').glob('batch_*/COMPLETED.json') if (p.parent/'classification_summary.json').is_file()]
    if source_output:
        source_output=Path(source_output).resolve()
        if source_output not in candidates:raise ValueError('Explicit source is not a completed result for this cell')
        source=source_output
    elif len(candidates)==1:source=candidates[0]
    else:raise ValueError('Expected one completed source; use --source-output to resolve multiple results')
    release=Path((CONTROL/'CURRENT_RELEASE.txt').read_text().strip())
    if not release.is_absolute():release=CONTROL/'releases'/release
    return cell,source,release

def prepare(args):
    cell,source,release=discover(args.cell,args.source_output)
    job=read(source/'job.json');old=read(source/'classification_summary.json');meta=read(source/'masks/metadata.json')
    if args.radii_px!=[1,2]:raise ValueError('This comparison uses exactly --radii-px 1 2')
    if job['sample_id']!='WT_TRAK2_BR1_T5':raise ValueError('This pilot is restricted to WT_TRAK2_BR1_T5')
    if any(old.get(k)!=v for k,v in BASELINE.items()):raise ValueError('Saved T5 baseline differs')
    if meta.get('recipe')!=RECIPE or meta.get('shape')!=[50,684,428] or meta.get('original_frame_to_block')!='f//50':raise ValueError('Unexpected mask recipe, dimensions, or mapping')
    if meta.get('pixel_size_nm')!=117 or job.get('frame_interval_s')!=.02:raise ValueError('Unexpected scientific scale')
    run_id=args.run_id or stamp()
    if not re.fullmatch(r'[A-Za-z0-9_]+',run_id):raise ValueError('Unsafe run ID')
    output=guard(cell/'analysis_results'/('distance_sensitivity_'+run_id))
    figures=guard(cell/('Analysis_Figures_distance_sensitivity_'+run_id))
    output.mkdir(parents=True,exist_ok=True);figures.mkdir(exist_ok=True);(output/'data').mkdir(exist_ok=True);(output/'checkpoints').mkdir(exist_ok=True);(output/'validation').mkdir(exist_ok=True)
    env_for(output)
    source_files={
      'localizations':str(source/'data/localizations.csv.gz'),
      'full_mask':str(source/'masks/clean_full_mask.npy'),
      'ROI_mask':str(source/'masks/analysis_ROI_mask.npy'),
      'roi':str(source/'roi/roi_mask.npy'),
      'vertices':str(source/'roi/vertices.npy'),
      'frame_mapping':str(source/'masks/original_frame_to_mask_block.npy'),
      'C1':str(cell/Path(job['C1_file']).name),'C2':str(cell/Path(job['C2_file']).name),
      'ROI_file':str(cell/Path(job['ROI_file']).name)}
    methods=[dict(key='old_k1',label='Original k=1 SE ellipse',field='pointwise_label'),
      dict(key='close_1px',label='1-pixel close (117 nm)',field='close_1px_label'),
      dict(key='close_2px',label='2-pixel close (234 nm)',field='close_2px_label')]
    protect=list(source_files.values())+[str(source/f) for f in ['COMPLETED.json','classification_summary.json','masks/metadata.json','masks/source_binding.json','classification_source_binding.json','job.json']]+[str(CONTROL/'CURRENT_RELEASE.txt'),str(release/'vendor/spatial_mask_preview/geometry.py')]
    before=[descriptor(p) for p in protect]
    config=dict(schema_version=1,run_id=run_id,sample_id=job['sample_id'],sample_name=job['sample_name'],
      cell=str(cell),source_output=str(source),source_release=str(release),output=str(output),output_directory=str(output),
      figures=str(figures),methods=methods,radii_px=args.radii_px,radii_nm=[117,234],pixel_size_nm=117,frame_interval_s=.02,
      frames=2500,mask_block_frames=50,mask_recipe=RECIPE,crop_xyxy_px=job['crop_xyxy_px'],
      sample_frames=SAMPLE_FRAMES,geometry='Euclidean proximity to union of closed foreground pixel squares',
      tangency_included=True,q_tolerance=1e-12,classification_mask='full, before ROI clipping',QC_selection='original QC AND refined center inside original ROI',
      unknown_policy='radius-specific raw same-original-frame C1 zero-square contact or extension beyond closed FOV',
      distance_radii_are='geometric tolerances, not estimated localization SE or mask confidence regions',
      registration_status=old['registration_status'],calibration_status='manufacturer_nominal',
      source_files=source_files,baseline=BASELINE,source_fingerprints_before=before,
      new_TARDIS_calculations=0,new_detection_fits_or_segmentation=0)
    signature=hashlib.sha256(json.dumps(dict(sources=before,radii=args.radii_px,geometry_sha256=hash_file(release/'vendor/spatial_mask_preview/geometry.py')),sort_keys=True).encode()).hexdigest()
    config['classification_signature']=signature
    config_path=output/'config.json'
    if config_path.exists():
        existing=read(config_path)
        if existing['classification_signature']!=signature:raise ValueError('Run ID exists with changed sources/parameters')
    else:dump(config_path,config)
    return config

def classify(config):
    import numpy as np
    import tifffile
    release=Path(config['source_release']);sys.path.insert(0,str(release/'vendor/spatial_mask_preview'))
    import geometry
    out=Path(config['output']);files=config['source_files'];radii=config['radii_px']
    full=np.load(files['full_mask'],mmap_mode='r',allow_pickle=False)
    clipped=np.load(files['ROI_mask'],mmap_mode='r',allow_pickle=False)
    roi=np.load(files['roi'],allow_pickle=False);mapping=np.load(files['frame_mapping'],allow_pickle=False)
    if full.shape!=(50,684,428) or full.dtype!=np.bool_ or roi.shape!=(684,428):raise ValueError('Invalid masks')
    if not np.array_equal(mapping,np.arange(2500)//50):raise ValueError('Saved frame-to-block mapping differs')
    if not np.array_equal(clipped,full&roi[None,...]):raise ValueError('ROI mask is not full mask intersected with ROI')
    source_digest=hashlib.sha256();ids=set();uids=set();total=0;source_counts=Counter()
    parts=[]
    with gzip.open(files['localizations'],'rt',encoding='utf-8',newline='') as stream,tifffile.TiffFile(files['C1']) as tif:
        reader=csv.DictReader(stream);fields=reader.fieldnames
        if not fields or any(k in fields for k in new_fields(radii)):raise ValueError('Invalid or already extended source table')
        if len(tif.pages)!=2500 or tif.pages[0].shape!=(684,428) or tif.pages[0].dtype!=np.uint16:raise ValueError('Raw C1 dimensions/type differ')
        head=next(reader,None);previous=-1
        for start in range(0,2500,250):
            stop=start+250;rows=[]
            while head is not None and int(head['frame'])<stop:
                frame=int(head['frame'])
                if frame<start or frame<previous:raise ValueError('Source rows lost original frame ordering')
                previous=frame
                idx=head['candidate_index'];uid=head['source_detection_uid']
                if idx in ids or uid in uids:raise ValueError('Duplicate original candidate identity')
                ids.add(idx);uids.add(uid);row_digest(source_digest,head,fields)
                keep=selection(head)
                if keep!=flag(head['spatial_selected']):raise ValueError('Cached QC/ROI selection differs')
                label=head['pointwise_label']
                if keep:
                    if label not in LABELS:raise ValueError('Unexpected selected source label')
                    source_counts[DISPLAY[label]]+=1;source_counts['selected']+=1
                elif label!='excluded_qc':raise ValueError('Excluded source row does not retain excluded_qc')
                else:source_counts['QC_excluded']+=1
                source_counts['ROI_candidates']+=1;total+=1;rows.append(head);head=next(reader,None)
            part=out/'checkpoints'/f'rows_{start:04d}_{stop-1:04d}.csv.gz';checkpoint=part.with_suffix('.json')
            reusable=False
            if part.exists() and checkpoint.exists():
                state=read(checkpoint)
                if state.get('classification_signature')==config['classification_signature']:
                    with gzip.open(part,'rt',encoding='utf-8',newline='') as cached:oldrows=list(csv.DictReader(cached))
                    if len(oldrows)==len(rows) and all(all(a[k]==b[k] for k in fields) for a,b in zip(oldrows,rows)):reusable=True
            if reusable:event(config,'Reused classification checkpoint',frame_start=start,frame_stop_exclusive=stop,rows=len(rows))
            else:
                by_frame=defaultdict(list)
                for row in rows:by_frame[int(row['frame'])].append(row)
                temp=part.with_name(part.name+'.tmp')
                with gzip.open(temp,'wt',encoding='utf-8',newline='') as target:
                    writer=csv.DictWriter(target,fieldnames=fields+new_fields(radii));writer.writeheader()
                    for frame in range(start,stop):
                        raw=tif.pages[frame].asarray();zero=(raw==0);mask=full[int(mapping[frame])]
                        for row in by_frame[frame]:
                            extra={}
                            for radius in radii:
                                prefix=f'close_{radius}px_'
                                if selection(row):
                                    result=geometry.classify(mask,zero,number(row,'fit_x_px'),number(row,'fit_y_px'),float(radius),float(radius),1e-12)
                                    extra[prefix+'label']=result['label']
                                    extra[prefix+'unknown_reason']=result['reason'].replace('ellipse','distance_circle')
                                    extra[prefix+'intersects_known_mask']='' if result['intersects_known_mask'] is None else result['intersects_known_mask']
                                else:extra.update({prefix+'label':'excluded_qc',prefix+'unknown_reason':'',prefix+'intersects_known_mask':''})
                            writer.writerow(dict(row,**extra))
                os.replace(temp,part)
                dump(checkpoint,dict(complete=True,classification_signature=config['classification_signature'],frame_start=start,frame_stop_exclusive=stop,rows=len(rows),part=descriptor(part),completed_utc=utc()))
                event(config,'Classification checkpoint committed',frames_processed=stop,total_frames=2500,source_rows_seen=total,selected_seen=source_counts['selected'])
            parts.append(part)
        if head is not None:raise ValueError('Source contains frames after2499')
    for key,value in BASELINE.items():
        if source_counts[key]!=value:raise ValueError('Source classification baseline mismatch: '+key)
    original_digest=read(Path(config['source_output'])/'classification_summary.json')['original_field_strings_and_order_SHA256']
    # The historical digest covers pre-classification original fields; our digest
    # additionally protects every existing classification column in the current source.
    config['historical_original_fields_digest']=original_digest
    audit=out/'data/pointwise_comparison.csv.gz';temp=audit.with_name(audit.name+'.tmp')
    preserved_digest=hashlib.sha256();counts={m['key']:Counter() for m in config['methods']};common_on=Counter();common_count=0
    transitions=defaultdict(Counter);pairs=list(combinations(config['methods'],2));frame_stats=[]
    for frame in range(2500):
        rec=dict(frame=frame,time_s=frame*.02,ROI_candidates=0,selected=0,QC_excluded=0,common_known_count=0)
        for method in config['methods']:
            for name in ['On','Not_on','unknown']:rec[method['key']+'_'+name]=0
        frame_stats.append(rec)
    rows_written=0
    with gzip.open(temp,'wt',encoding='utf-8',newline='') as target:
        writer=csv.DictWriter(target,fieldnames=fields+new_fields(radii));writer.writeheader()
        for part in parts:
            with gzip.open(part,'rt',encoding='utf-8',newline='') as source:
                for row in csv.DictReader(source):
                    row_digest(preserved_digest,row,fields);writer.writerow(row);rows_written+=1
                    rec=frame_stats[int(row['frame'])];rec['ROI_candidates']+=1
                    keep=selection(row)
                    if not keep:
                        rec['QC_excluded']+=1
                        for method in config['methods']:
                            if row[method['field']]!='excluded_qc':raise ValueError('QC excluded point recovered')
                            counts[method['key']]['QC_excluded']+=1
                        continue
                    rec['selected']+=1;labels={m['key']:row[m['field']] for m in config['methods']}
                    for key,label in labels.items():
                        if label not in LABELS:raise ValueError('Unknown label value')
                        counts[key][DISPLAY[label]]+=1;counts[key]['selected']+=1;rec[key+'_'+DISPLAY[label]]+=1
                    known=all(label in LABELS[:2] for label in labels.values())
                    if known:
                        common_count+=1;rec['common_known_count']+=1
                        for key,label in labels.items():common_on[key]+=int(label==LABELS[0])
                        if labels['close_1px']==LABELS[0] and labels['close_2px']!=LABELS[0]:raise ValueError('Common-known circles violate nesting')
                    if flag(row['close_1px_intersects_known_mask']) and not flag(row['close_2px_intersects_known_mask']):raise ValueError('Raw mask hits violate radius nesting')
                    if labels['close_1px']=='unknown' and labels['close_2px']!='unknown':raise ValueError('Unknown sets violate radius nesting')
                    for left,right in pairs:transitions[(left['key'],right['key'])][(labels[left['key']],labels[right['key']])]+=1
    if source_digest.hexdigest()!=preserved_digest.hexdigest() or rows_written!=total:raise ValueError('Source fields changed in comparison audit')
    os.replace(temp,audit)
    summaries=[]
    for method in config['methods']:
        n=counts[method['key']];denom=n['On']+n['Not_on']
        if n['selected']!=BASELINE['selected'] or n['QC_excluded']!=BASELINE['QC_excluded'] or sum(n[k] for k in ['On','Not_on','unknown'])!=n['selected']:raise ValueError('Method quantity conservation failed')
        summaries.append(dict(method,**{k:n[k] for k in ['On','Not_on','unknown','QC_excluded','selected']},
          On_percentage=100*n['On']/denom if denom else None,on_denominator=denom,
          common_known_On_percentage=100*common_on[method['key']]/common_count if common_count else None,common_known_count=common_count))
    transition_rows=[];flips=[]
    for left,right in pairs:
        matrix=transitions[(left['key'],right['key'])]
        if sum(matrix.values())!=BASELINE['selected']:raise ValueError('Transition matrix conservation failed')
        for a in LABELS:
            for b in LABELS:transition_rows.append(dict(from_method=left['key'],to_method=right['key'],from_label=a,to_label=b,count=matrix[(a,b)]))
        flips.append(dict(comparison=left['label']+' → '+right['label'],from_method=left['key'],to_method=right['key'],
          On_to_Not_on=matrix[(LABELS[0],LABELS[1])],Not_on_to_On=matrix[(LABELS[1],LABELS[0])],
          other_transition_count=sum(value for (a,b),value in matrix.items() if a!=b and 'unknown' in (a,b)),
          common_known_count=sum(value for (a,b),value in matrix.items() if a!='unknown' and b!='unknown'),
          unchanged=sum(value for (a,b),value in matrix.items() if a==b)))
    for rec in frame_stats:
        if rec['selected']+rec['QC_excluded']!=rec['ROI_candidates']:raise ValueError('Frame candidate conservation failed')
        for method in config['methods']:
            if sum(rec[method['key']+'_'+name] for name in ['On','Not_on','unknown'])!=rec['selected']:raise ValueError('Frame class conservation failed')
    write_csv(out/'data/per_frame_counts.csv',frame_stats,list(frame_stats[0]))
    write_csv(out/'transitions.csv',transition_rows,['from_method','to_method','from_label','to_label','count'])
    write_csv(out/'method_counts.csv',summaries,['key','label','field','On','Not_on','unknown','QC_excluded','selected','On_percentage','on_denominator','common_known_On_percentage','common_known_count'])
    write_csv(out/'classification_flips.csv',flips,['comparison','from_method','to_method','On_to_Not_on','Not_on_to_On','other_transition_count','common_known_count','unchanged'])
    summary=dict(sample_id=config['sample_id'],methods=summaries,flips=flips,ROI_candidates=total,selected=BASELINE['selected'],QC_excluded=BASELINE['QC_excluded'],common_known_count=common_count,
       source_table_all_fields_SHA256=source_digest.hexdigest(),saved_source_field_prefix_SHA256=preserved_digest.hexdigest(),
       original_fields_and_rows_unchanged=True,original_fieldnames=fields,new_fields=new_fields(radii),source_sample_frames=SAMPLE_FRAMES,
       scientific_interpretation='Fixed radii are geometric tolerances; spatial proximity is not proof of binding.',new_TARDIS_calculations=0)
    dump(out/'summary.json',summary)
    dump(out/'CLASSIFIED.json',dict(complete=True,classification_signature=config['classification_signature'],rows=total,source_fields_SHA256=source_digest.hexdigest(),audit=descriptor(audit),completed_utc=utc()))
    event(config,'All 2500 frames classified',methods=summaries)
    return summary

def run_child(config,command,label):
    event(config,label+' started',command=command)
    with (Path(config['output'])/'run.log').open('a',encoding='utf-8') as stream:
        process=subprocess.Popen(command,cwd=HERE,env=dict(os.environ),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in process.stdout:print(line,end='',flush=True);stream.write(line);stream.flush()
        code=process.wait()
    if code:raise RuntimeError(label+' failed with exit '+str(code))
    event(config,label+' completed')

def finalize(config):
    out=Path(config['output']);after=[descriptor(record['path']) for record in config['source_fingerprints_before']]
    if after!=config['source_fingerprints_before']:raise ValueError('Protected source fingerprints changed')
    dump(out/'source_fingerprints.json',dict(before=config['source_fingerprints_before'],after=after,all_unchanged=True))
    snapshot=out/'source';snapshot.mkdir(exist_ok=True)
    copies=[HERE/'compare_assignment_distance.py',HERE/'distance_plots.py',HERE/'reader_workbook.mjs',HERE/'render_mean_overlays.py',
      Path(config['source_release'])/'vendor/spatial_mask_preview/geometry.py',
      HERE/'tests/reference_geometry.py',HERE/'tests/test_fixed_geometry.py',HERE/'tests/validate_t5_pilot.py']
    for source in copies:
        if source.exists():shutil.copyfile(source,snapshot/source.name)
    codefiles=[dict(path=p.name,sha256=hash_file(p)) for p in sorted(snapshot.iterdir()) if p.is_file()]
    versions=dict(python=sys.version,executable=sys.executable,packages={})
    for name in ['numpy','scipy','tifffile','shapely','pandas','Pillow','matplotlib']:
        try:versions['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    runtime=read(Path(config['source_release'])/'runtime_paths.json')
    version_code='import importlib.metadata as m,json,sys;print(json.dumps(dict(python=sys.version,executable=sys.executable,packages={n:m.version(n) for n in ["numpy","pandas","Pillow","matplotlib","tifffile"]})))'
    versions['plotting_environment']=json.loads(subprocess.check_output([runtime['science_python'],'-B','-s','-c',version_code],text=True,env=dict(os.environ)))
    artifact_package=Path('/Users/gaoxinran/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/package.json')
    if artifact_package.exists():versions['spreadsheet_artifact_tool_version']=read(artifact_package).get('version')
    dump(out/'versions.json',versions);dump(out/'source_manifest.json',dict(files=codefiles))
    with (out/'README.txt').open('w',encoding='utf-8') as stream:
        stream.write('WT TRAK2 BR1 T5 assignment-distance sensitivity\n\nOriginal SE ellipse labels and fixed1/2pixel geometric tolerance labels use identical source points,QC,ROI,and full block masks. C1/C2 identity registration remains unverified.\nUnknown and QC exclusions remain separate. On percentage = On/(On+Not-on). Fixed circles do not replace measured localization SE. No TARDIS,tracking,detection,fitting,or segmentation was run.\n\nFigures: '+config['figures']+'\nReader workbook: Distance_Comparison.xlsx\nAll original source fields are preserved in data/pointwise_comparison.csv.gz. Independent complete foreground enumeration covers ten representative frames only; this is not a full-series independent geometry validation.\n')
    validation=read(out/'validation/independent_validation.json')
    if validation.get('all_passed') is not True:raise ValueError('Independent validation did not pass')
    plot=read(out/'plot_manifest.json')
    if len(plot['files'])!=14 or not all((Path(config['figures'])/name).is_file() for name in plot['files']):raise ValueError('Required figures are incomplete')
    if not (out/'Distance_Comparison.xlsx').is_file():raise ValueError('Reader workbook is absent')
    # Retain checkpoint records; discard only our now-redundant temporary row
    # shards after the complete canonical audit and validation are committed.
    removed=0
    for state_path in sorted((out/'checkpoints').glob('rows_*.csv.json')):
        state=read(state_path);payload=Path(state['part']['path'])
        if payload.exists():payload.unlink();removed+=1
        state.update(payload_status='merged_into_final_audit_and_cleaned',final_audit_relative='data/pointwise_comparison.csv.gz')
        dump(state_path,state)
    with (CONTROL/'BATCH_LOG.txt').open('a',encoding='utf-8') as stream:
        stream.write('\n'+utc()+' | Completed isolated WT_TRAK2_BR1_T5 assignment-distance pilot (old k1 vs fixed1px/2px); original outputs preserved; no TARDIS or additional cells started. Results: '+str(out)+'\n')
    event(config,'Pilot complete; stopped after T5')
    failed=out/'FAILED.json'
    if failed.exists():failed.unlink()
    dump(out/'COMPLETE.json',dict(complete=True,sample_id=config['sample_id'],run_id=config['run_id'],completed_utc=utc(),frames=2500,classified_rows=BASELINE['ROI_candidates'],
      input_fingerprints_unchanged=True,original_fields_unchanged=True,independent_check_frames=SAMPLE_FRAMES,
      classification_validation='validation/independent_validation.json',figures=config['figures'],reader_workbook='Distance_Comparison.xlsx',
      checkpoint_payloads_cleaned=removed,no_TARDIS=True,no_additional_cells=True))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell',type=Path,required=True)
    parser.add_argument('--radii-px',type=int,nargs='+',default=[1,2])
    parser.add_argument('--run-id')
    parser.add_argument('--source-output',type=Path)
    parser.add_argument('--classify-only',action='store_true',help='Prepare classifications/checkpoints only; do not mark final completion')
    args=parser.parse_args()
    config=None
    try:
        config=prepare(args);out=Path(config['output']);config_path=out/'config.json';runtime=read(Path(config['source_release'])/'runtime_paths.json')
        dump(out/'command.json',dict(argv=[sys.executable]+sys.argv,cwd=str(HERE),started_utc=utc()))
        event(config,'Starting isolated assignment-distance pilot')
        if (out/'CLASSIFIED.json').exists() and read(out/'CLASSIFIED.json')['classification_signature']==config['classification_signature']:
            event(config,'Reusing completed new classification audit')
        else:classify(config)
        if args.classify_only:return 0
        sys.path.insert(0,str(HERE))
        if not (out/'plot_manifest.json').exists():
            command=[runtime['science_python'],'-B','-s','-c','import sys;sys.path.insert(0,sys.argv[1]);from distance_plots import render_outputs;render_outputs(sys.argv[2])',str(HERE),str(config_path)]
            run_child(config,command,'Static comparison figures')
        if not (out/'MEAN_OVERLAYS_COMPLETE.json').exists():
            run_child(config,[runtime['science_python'],'-B','-s',str(HERE/'render_mean_overlays.py'),'--config',str(config_path)],'Not-on overlays on cached C1 mean')
        if not (out/'Distance_Comparison.xlsx').exists():
            node=Path('/Users/gaoxinran/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
            if not node.is_file():raise RuntimeError('Existing spreadsheet runtime unavailable')
            run_child(config,[str(node),str(HERE/'reader_workbook.mjs'),str(config_path)],'Reader workbook')
        validator=HERE/'tests/validate_t5_pilot.py'
        if not validator.is_file():raise RuntimeError('Independent ten-frame validator is not installed')
        run_child(config,[runtime['cleanup_python'],'-B','-s',str(HERE/'tests/test_fixed_geometry.py')],'Synthetic geometry checks')
        run_child(config,[runtime['cleanup_python'],'-B','-s',str(validator),'--run',str(out)],'Independent ten-frame validation')
        finalize(config);return 0
    except Exception as error:
        if config:
            dump(Path(config['output'])/'FAILED.json',dict(utc=utc(),error=repr(error),complete=False))
            event(config,'Pilot stopped with error',error=repr(error))
        raise
if __name__=='__main__':raise SystemExit(main())
