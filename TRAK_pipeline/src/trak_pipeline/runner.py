"""Dependency-aware execution and strictly non-computing refresh."""
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
import concurrent.futures,json,os,sys,threading,time,traceback
from .types import Artifact,StageContext,StageResult
from .graph import default_specs,stage_parameters,build_plan,expected_identity
from .cache import CacheStore
from .legacy import LegacyReader,source_artifacts
from .identity import digest
from .io import atomic_json,read_json,safe,guard,utc
from . import runtime
from .config import portable_input

ROLES={'mask_cleanup':'cleanup_python','mask_roi':'cleanup_python','assignment':'cleanup_python'}
_EVENT_LOCK=threading.Lock()

def method_family(config):
    return digest({s.stage:dict(method_id=s.method_id,method_version=s.method_version,parameters=stage_parameters(config,s))for s in default_specs(config)if s.scientific})

def _state(config):return Path(config['control_root'])/'modular/state'
def _event(config,event):
    directory=guard(_state(config),config['data_root']);directory.mkdir(parents=True,exist_ok=True)
    with _EVENT_LOCK,(directory/'events.jsonl').open('a',encoding='utf-8')as stream:stream.write(json.dumps(safe(dict(utc=utc(),**event)))+'\n')

def _specs(config,target=None):
    specs=default_specs(config)
    if not target:return specs
    by={s.stage:s for s in specs}
    if target not in by:raise ValueError('Unknown stage target: '+target)
    needed=set()
    def add(stage):
        if stage in needed:return
        needed.add(stage)
        if stage in by:
            for d in by[stage].dependencies:add(d)
    add(target)
    return[s for s in specs if s.stage in needed]

def _attach_cached_roi(config,sample,artifact):
    """Read an existing raster cache without computing geometry during refresh."""
    if all(k in artifact.files and Path(artifact.files[k]).is_file()for k in('mask','vertices')):return artifact
    marker=sample.directory/'analysis_results/modular_cache/roi_geometry'/artifact.id/'geometry.json'
    if not marker.is_file():return artifact
    saved=read_json(marker)
    if saved.get('source_roi_id',artifact.id)!=artifact.id:return artifact
    files={k:str(portable_input(v,config))for k,v in saved.get('files',{}).items()}
    if all(k in files and Path(files[k]).is_file()for k in('mask','vertices')):
        artifact.files.update(files);artifact.metadata.update(saved.get('metadata',{}))
    return artifact

def available_artifacts(config,sample,reader=None,validate=False):
    reader=reader or LegacyReader(config)
    artifacts=reader.read_sample(sample,validate=validate)
    if 'roi'in artifacts:_attach_cached_roi(config,sample,artifacts['roi'])
    store=CacheStore(config,sample);available={k:v for k,v in artifacts.items()if k in('C1','C2','roi')}
    for spec in default_specs(config):
        if any(d not in available for d in spec.dependencies):continue
        inputs={d:available[d]for d in spec.dependencies};params=stage_parameters(config,spec);key=expected_identity(spec,params,inputs)
        prior=artifacts.get(spec.stage)
        match=store.lookup(spec,params,inputs)or(prior if prior and prior.id==key else None)
        if match is not None:available[spec.stage]=match
    return available

def _ensure_roi(config,sample,artifact,counter):
    if all(k in artifact.files and Path(artifact.files[k]).is_file()for k in('mask','vertices')):return artifact
    from .stages.localization import execute_roi_geometry
    folder=sample.directory/'analysis_results/modular_cache/roi_geometry'/artifact.id
    marker=folder/'geometry.json'
    _attach_cached_roi(config,sample,artifact)
    if all(k in artifact.files and Path(artifact.files[k]).is_file()for k in('mask','vertices')):return artifact
    ctx=StageContext(config,sample,{},dict(roi=artifact),folder,Path(config['package_root']),config['runtime'])
    counter['roi_geometry']=counter.get('roi_geometry',0)+1;_event(config,dict(event='scientific_stage_call',stage='roi_geometry',sample_id=sample.sample_id))
    result=execute_roi_geometry(ctx);artifact.files.update({k:str(v)for k,v in result.files.items()});artifact.metadata.update(result.metadata)
    atomic_json(marker,dict(source_roi_id=artifact.id,files={k:Path(v).resolve().relative_to(Path(config['data_root']).resolve()).as_posix()for k,v in artifact.files.items()},metadata=safe(artifact.metadata)),config['data_root']);return artifact

def _execute(config,sample,spec,params,inputs,store,counter):
    cache_dir=guard(store.output_dir(spec,params,inputs),config['data_root']);cache_dir.mkdir(parents=True,exist_ok=True)
    output=cache_dir
    if not spec.scientific:
        output=sample.directory/('Analysis_Figures_modular_'+cache_dir.name[:12]);guard(output,config['data_root']);output.mkdir(parents=True,exist_ok=True)
    ctx=StageContext(config,sample,params,inputs,output,Path(config['package_root']),config.get('runtime',{}))
    request=safe(asdict(ctx));request.update(stage=spec.stage,result_path=str(cache_dir/'stage_result.json'))
    path=cache_dir/'request.json';atomic_json(path,request,config['data_root'])
    role=ROLES.get(spec.stage,'science_python');python=config.get('runtime',{}).get(role)
    if not python or not Path(python).is_file():raise runtime.platform.BatchInfrastructureError('Required installed runtime is missing: '+role)
    with _EVENT_LOCK:counter[spec.stage]=counter.get(spec.stage,0)+int(spec.scientific)
    _event(config,dict(event='scientific_stage_call'if spec.scientific else'presentation_call',stage=spec.stage,sample_id=sample.sample_id,artifact_id=cache_dir.name))
    print(f'{sample.sample_id} | {spec.stage} | '+('compute'if spec.scientific else'publish'),flush=True)
    runtime.invoke(config,[python,'-B','-s','-u',str(Path(config['package_root'])/'tools/worker_bootstrap.py'),'--request',str(path)],cache_dir/'stage.log')
    payload=read_json(cache_dir/'stage_result.json');result=StageResult(payload['files'],payload.get('metadata',{}))
    return store.commit(spec,params,inputs,result)

def run_sample(config,sample,reader=None,validate=False,target=None,presentation_only=False,force_stages=()):
    runtime.reset_cancellation()
    reader=reader or LegacyReader(config);prior=reader.read_sample(sample,validate=validate)
    if 'roi'in prior:_attach_cached_roi(config,sample,prior['roi'])
    available={k:v for k,v in prior.items()if k in('C1','C2','roi')};store=CacheStore(config,sample)
    specs=_specs(config,target);by={s.stage:s for s in default_specs(config)};counter={};actions=[];visiting=set();future_broad=None;pool=None;publication_error=None
    def resolve_broad():
        # Nellie runs on its own thread; only stages that depend on broad wait for it.
        if future_broad is not None and 'broad' not in available:
            available['broad']=future_broad.result();actions.append(dict(stage='broad',action='compute',artifact_id=available['broad'].id))
    def ensure(stage,materialize=False):
        if stage=='broad':resolve_broad()
        if stage in('C1','C2','roi'):
            if stage not in available:raise FileNotFoundError('Source artifact missing: '+stage)
            if stage=='roi'and materialize:available[stage]=_ensure_roi(config,sample,available[stage],counter)
            return available[stage]
        if stage in available and not(materialize and available[stage].metadata.get('materialization_required')):return available[stage]
        if stage in visiting:raise RuntimeError('Dependency cycle at '+stage)
        visiting.add(stage)
        try:
            spec=by[stage];inputs={d:ensure(d)for d in spec.dependencies};params=stage_parameters(config,spec);key=expected_identity(spec,params,inputs)
            old=prior.get(stage);match=None if stage in force_stages else(store.lookup(spec,params,inputs)or(old if old and old.id==key else None))
            if match is not None and not(materialize and match.metadata.get('materialization_required')):
                available[stage]=match;actions.append(dict(stage=stage,action='reuse',artifact_id=match.id));return match
            if presentation_only and spec.scientific:
                raise FileNotFoundError('Missing scientific artifact '+stage+'; refresh cannot compute it. Use an explicit run/summarize command.')
            inputs={d:ensure(d,materialize=True)for d in spec.dependencies}
            result=_execute(config,sample,spec,params,inputs,store,counter)
            available[stage]=result;actions.append(dict(stage=stage,action='compute'if spec.scientific else'publish',artifact_id=result.id));return result
        finally:visiting.remove(stage)
    # Independent heavy stages can run concurrently. All writes remain separately
    # owned. Cancellation terminates the central child-process registry first.
    if not presentation_only and config.get('execution',{}).get('parallel_independent_stages',True)and any(s.stage=='broad'for s in specs):
        bspec=by['broad'];bins={'C1':available['C1']};bp=stage_parameters(config,bspec);bk=expected_identity(bspec,bp,bins)
        old=prior.get('broad');bm=None if 'broad'in force_stages else(store.lookup(bspec,bp,bins)or(old if old and old.id==bk else None))
        if bm is not None:available['broad']=bm
        else:
            pool=concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future_broad=pool.submit(_execute,config,sample,bspec,bp,bins,store,counter)
    try:
        for spec in specs:
            if spec.stage=='broad'and future_broad is not None:
                continue  # deferred: fit and other broad-independent stages proceed meanwhile
            if not spec.scientific:
                try:ensure(spec.stage)
                except runtime.platform.BatchInfrastructureError:raise
                except Exception as error:
                    if isinstance(error,OSError)and(getattr(error,'errno',None)in(5,6,19,28,30,57,60,74,121,122)or not Path(config['data_root']).is_dir()):raise
                    publication_error=dict(stage=spec.stage,error=str(error),traceback=traceback.format_exc())
                    _event(config,dict(event='publication_failed',sample_id=sample.sample_id,**publication_error))
            else:ensure(spec.stage)
        resolve_broad()
    except BaseException:
        runtime.terminate_children()
        raise
    finally:
        if pool:pool.shutdown(wait=True,cancel_futures=True)
    report=dict(sample_id=sample.sample_id,method_family=method_family(config),completed_utc=utc(),actions=actions,scientific_stage_calls=sum(counter.values()),stage_calls=counter,artifacts={k:safe(asdict(v))for k,v in available.items()},legacy_validation=reader.reports.get(sample.sample_id),target=target,science_complete=all(s.stage in available for s in default_specs(config)if s.scientific),publication_complete='presentation'in available,publication_error=publication_error)
    state=_state(config)/'cells'/sample.sample_id/(report['method_family']+'.json');atomic_json(state,report,config['data_root'])
    return report,available


def run_dataset(config,samples,validate=False,target=None,skip_summary=False,resume=False,force_stages=()):
    runtime.configure(config);runtime.validate_runtime(config);state=_state(config);stop=state/'STOP_AFTER_CURRENT_CELL'
    if resume:
        for name in('STOP_AFTER_CURRENT_CELL','STOP_REQUESTED'):
            p=state/name
            if p.exists():os.replace(p,p.with_name(p.name+'.cleared_'+str(time.time_ns())))
    if stop.exists():raise RuntimeError('Stop-after-cell is pending. Use the explicit Resume launcher.')
    from .config import discover_samples
    summary_samples=discover_samples(config)if not skip_summary and target is None else samples
    reader=LegacyReader(config);report=dict(started_utc=utc(),method_family=method_family(config),cells=[],new_cells_started=0,scientific_stage_calls=0)
    try:
        if not skip_summary and target is None:
            try:report['startup_summary']=summarize_dataset(config,summary_samples,publish=True)
            except runtime.platform.BatchInfrastructureError:raise
            except OSError as error:
                if getattr(error,'errno',None)in(5,6,19,28,30,57,60,74,121,122)or not Path(config['data_root']).is_dir():raise
                report.setdefault('summary_failures',[]).append(dict(phase='startup',error=str(error)))
            except Exception as error:report.setdefault('summary_failures',[]).append(dict(phase='startup',error=str(error)))
        for sample in samples:
            try:
                row,artifacts=run_sample(config,sample,reader,validate,target,force_stages=force_stages);report['cells'].append(dict(sample_id=sample.sample_id,status='complete'if row['publication_complete']or target else'publication_incomplete',scientific_stage_calls=row['scientific_stage_calls'],science_complete=row['science_complete'],publication_complete=row['publication_complete']))
                report['scientific_stage_calls']+=row['scientific_stage_calls'];report['new_cells_started']+=int(row['scientific_stage_calls']>0)
                if not skip_summary and target is None and row['scientific_stage_calls']>0:
                    # Unchanged reused cells are already covered by startup. Aggregation is explicit science, publication is separate.
                    try:summarize_dataset(config,summary_samples,publish=True)
                    except runtime.platform.BatchInfrastructureError:raise
                    except OSError as error:
                        if getattr(error,'errno',None)in(5,6,19,28,30,57,60,74,121,122)or not Path(config['data_root']).is_dir():raise
                        report.setdefault('summary_failures',[]).append(dict(after_sample=sample.sample_id,error=str(error),traceback=traceback.format_exc()))
                    except Exception as error:
                        report.setdefault('summary_failures',[]).append(dict(after_sample=sample.sample_id,error=str(error),traceback=traceback.format_exc()))
            except(runtime.platform.BatchInfrastructureError,OSError)as error:
                if isinstance(error,runtime.platform.BatchInfrastructureError)or getattr(error,'errno',None)in(5,6,19,28,30,57,60,74,121,122)or not Path(config['data_root']).is_dir():raise
                report['cells'].append(dict(sample_id=sample.sample_id,status='failed',error=str(error),traceback=traceback.format_exc()))
            except Exception as error:
                report['cells'].append(dict(sample_id=sample.sample_id,status='failed',error=str(error),traceback=traceback.format_exc()))
            atomic_json(state/'active_run.json',report,config['data_root'])
            if stop.exists():report['stopped_after_cell']=sample.sample_id;break
    finally:
        runtime.terminate_children();report['ended_utc']=utc();atomic_json(state/'last_run.json',report,config['data_root'])
    return report


def _summary_records(config,samples):
    reader=LegacyReader(config);records=[];missing=[]
    for sample in samples:
        artifacts=available_artifacts(config,sample,reader,False)
        if all(k in artifacts for k in('fit','assignment','mask_cleanup','mask_roi','decomposition')):records.append(dict(sample=sample,artifacts=artifacts))
        else:missing.append(dict(sample_id=sample.sample_id,missing=[k for k in('fit','assignment','mask_cleanup','mask_roi','decomposition')if k not in artifacts]))
    return records,missing

def summarize_dataset(config,samples,publish=True):
    from .stages import summaries
    records,missing=_summary_records(config,samples)
    if not records:return dict(completed_cells=0,missing=missing,scientific_stage_calls=0,status='no_completed_scientific_products')
    summary_config=dict(config,_expected_samples=[dict(sample.metadata,sample_id=sample.sample_id,source_directory=str(sample.directory),source_directory_relative=sample.directory.relative_to(Path(config['data_root'])).as_posix())for sample in samples])
    result=summaries.summarize(summary_config,records)
    calls=0 if isinstance(result,dict)and result.get('reused')else 1
    if calls:_event(config,dict(event='scientific_stage_call',stage='hierarchy_summary',completed_cells=len(records)))
    if publish:summaries.refresh(config,result.get('manifest_path',result.get('manifest'))if isinstance(result,dict)else result)
    return dict(result=result,missing=missing,completed_cells=len(records),scientific_stage_calls=calls)


def refresh_dataset(config,samples):
    """Read/paint/export only. Missing science never schedules ensure() execution."""
    runtime.reset_cancellation();runtime.configure(config);reader=LegacyReader(config);report=dict(scientific_stage_calls=0,new_cells_started=0,cells=[],missing_scientific_artifacts=[])
    for sample in samples:
        try:
            artifacts=available_artifacts(config,sample,reader,False)
            if 'presentation'in artifacts:
                report['cells'].append(dict(sample_id=sample.sample_id,status='reused',figures=artifacts['presentation'].files));continue
            specs={s.stage:s for s in default_specs(config)};spec=specs['presentation'];absent=[d for d in spec.dependencies if d not in artifacts or artifacts[d].metadata.get('materialization_required')]
            if absent:raise FileNotFoundError('Missing saved scientific products: '+','.join(absent))
            store=CacheStore(config,sample);params=stage_parameters(config,spec);inputs={d:artifacts[d]for d in spec.dependencies};counter={}
            result=_execute(config,sample,spec,params,inputs,store,counter)
            if sum(counter.values())!=0:raise AssertionError('Refresh dispatched science')
            report['cells'].append(dict(sample_id=sample.sample_id,status='published',figures=result.files))
        except Exception as e:report['missing_scientific_artifacts'].append(dict(sample_id=sample.sample_id,error=str(e)))
    from .stages import summaries
    try:report['hierarchy']=summaries.refresh(config)
    except FileNotFoundError as e:report['missing_scientific_artifacts'].append(dict(stage='hierarchy_summary',error=str(e)))
    report['complete']=not report['missing_scientific_artifacts'];atomic_json(_state(config)/'last_refresh.json',report,config['data_root'])
    return report
