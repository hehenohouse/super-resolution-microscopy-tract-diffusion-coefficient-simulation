#!/usr/bin/env python3
"""Pinned, resumable BR4/BR5 queue; shared science, all-five-BR summaries.

This controller never implements detection, segmentation or TARDIS. Cell work
is restricted to manifest.new_sample_ids; summaries receive the full inventory.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import traceback
import uuid

INFRA_ERRNOS = {5,6,19,28,30,57,60,74,121,122}


def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def digest(value): return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8');os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def manifest_paths(path,require_ready=True):
    path=Path(path).resolve();m=read(path);batch=path.parent
    common=(batch/m.get('common_root_relative','../..')).resolve()
    if m.get('schema_version')!=1 or m.get('path_basis')!='common_root':raise ValueError('Unsupported batch manifest')
    if not batch.is_relative_to(common/'batches'):raise ValueError('Batch must remain inside the shared package batches directory')
    if require_ready and (not m.get('ready_for_execution') or not m.get('release_relative')):
        raise ValueError('This batch has not been frozen and marked ready; no cell was started')
    molecules=[d.get('molecule') for d in m.get('datasets',[])]
    if molecules!=['TRAK1','TRAK2']:raise ValueError('Expected explicit TRAK1 then TRAK2 datasets')
    all_ids=[]
    for d in m['datasets']:
        ids=d['new_sample_ids'];all_ids.extend(ids)
        if len(ids)!=len(set(ids)) or len(ids)!=d['new_cells']:raise ValueError('Invalid new-cell queue for '+d['molecule'])
        config=(common/d['config_relative']).resolve()
        if not config.is_relative_to(common) or not config.is_file():raise ValueError('Dataset configuration is missing or escapes the package')
    if len(all_ids)!=len(set(all_ids)) or len(all_ids)!=m['new_cells']:raise ValueError('Batch queue IDs must be globally unique')
    return m,common,batch


def early_environment(batch):
    for key,sub in [('TMPDIR','tmp'),('TMP','tmp'),('TEMP','tmp'),('MPLCONFIGDIR','matplotlib'),('NUMBA_CACHE_DIR','numba'),('XDG_CACHE_HOME','xdg')]:
        p=batch/'runtime'/sub;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',MPLBACKEND='Agg',PYTHONUTF8='1',PYTHONIOENCODING='utf-8')
    sys.dont_write_bytecode=True


def activate_pinned(common,manifest):
    release=(common/manifest['release_relative']).resolve()
    if not release.is_relative_to(common/'releases'):raise ValueError('Pinned release escapes the shared release directory')
    spec=importlib.util.spec_from_file_location('_extension_release_loader',common/'tools/launch.py')
    loader=importlib.util.module_from_spec(spec);spec.loader.exec_module(loader)
    loader.verify_release(release,read(release/'release_manifest.json'))
    os.environ.update(TRAK_COMMON_ROOT=str(common),TRAK_ACTIVE_RELEASE=str(release))
    sys.path.insert(0,str(release/'src'))
    return release


class SharedBackend:
    """Thin adapter to the frozen production runner; no numerical algorithms."""
    def __init__(self,common,manifest):
        from trak_pipeline import config,runtime,runner
        from trak_pipeline.legacy import source_artifacts
        self.api=config;self.runtime=runtime;self.runner=runner;self.sources=source_artifacts
        self.platform=runtime.platform;self.common=common;self.manifest=manifest
    def load_datasets(self):
        values=[]
        for record in self.manifest['datasets']:
            config=self.api.load_config(self.common/record['config_relative'])
            samples=self.api.discover_samples(config);byid={s.sample_id:s for s in samples}
            if len(samples)!=record['total_cells'] or len(byid)!=len(samples):raise ValueError('Combined inventory size/identity differs for '+record['molecule'])
            new=[]
            for sid in record['new_sample_ids']:
                if sid not in byid:raise ValueError('Queued cell is outside the combined inventory: '+sid)
                sample=byid[sid]
                if sample.metadata.get('BR') not in ('BR4','BR5'):raise ValueError('Old BR1-3 cell cannot enter the extension queue: '+sid)
                new.append(sample)
            if {s.metadata.get('BR') for s in samples}!={'BR1','BR2','BR3','BR4','BR5'}:raise ValueError('Five explicit replicas are required')
            values.append(dict(record=record,config=config,all_samples=samples,new_samples=new,molecule=record['molecule']))
        self.datasets=values
        return values
    def family(self,d):return self.runner.method_family(d['config'])
    def source_stamp(self,d,s):return {k:v.id for k,v in self.sources(d['config'],s).items()}
    def validate(self,d):
        self.runtime.configure(d['config']);self.runtime.validate_runtime(d['config'])
    def run_cell(self,d,s):
        # The only cell-entry call. skip_summary avoids a repeated startup
        # aggregation before every cell. Aggregation is explicit below.
        report=self.runner.run_dataset(d['config'],[s],skip_summary=True,resume=True)
        if not report.get('cells'):raise RuntimeError('Shared runner returned no cell record')
        row=report['cells'][0]
        return dict(row,shared_report_scientific_stage_calls=report.get('scientific_stage_calls',0))
    def completion_outputs_valid(self,d,s,previous):
        # Presence and recorded byte sizes only. No output hashing or pixel reads.
        try:
            from trak_pipeline.cache import resolve_portable
            config=d['config'];family=self.family(d)
            path=Path(config['control_root'])/'modular/state/cells'/s.sample_id/(family+'.json')
            report=read(path)
            if not report.get('science_complete') or not report.get('publication_complete'):return False
            for spec in self.runner.default_specs(config):
                art=report['artifacts'][spec.stage]
                marker=s.directory/'analysis_results/modular_cache'/spec.stage/art['id']/'artifact.json'
                committed=read(marker)
                if not committed.get('complete') or committed['artifact']['id']!=art['id']:return False
                for descriptor in committed['descriptors'].values():
                    target=resolve_portable(descriptor['location'],config['data_root'],config['package_root'])
                    if descriptor['type']=='file':
                        if not target.is_file() or target.stat().st_size!=descriptor['fingerprint']['bytes']:return False
                    elif descriptor['type']=='directory':
                        if not target.is_dir():return False
                        for rel,fp in descriptor['entries'].items():
                            child=target/rel
                            if not child.is_file() or child.stat().st_size!=fp['bytes']:return False
                    else:return False
            return True
        except (OSError,ValueError,KeyError,TypeError):return False
    def summarize(self,d,failed_ids=()):
        config=dict(d['config'],summary_failed_sample_ids=list(failed_ids))
        return self.runner.summarize_dataset(config,d['all_samples'],publish=True)
    def refresh(self,d):return self.runner.refresh_dataset(d['config'],[])
    def infrastructure(self,error):
        return isinstance(error,self.platform.BatchInfrastructureError) or isinstance(error,ImportError) or (isinstance(error,OSError) and (error.errno in INFRA_ERRNOS or any(not Path(d['config']['data_root']).is_dir() for d in self.datasets) or not os.path.ismount(str(self.common.parent))))
    def terminate(self):self.runtime.terminate_children()
    def lock_contexts(self,datasets):
        stack=ExitStack()
        try:
            for d in datasets:
                config=d['config'];self.runtime.configure(config)
                # Honor the ordinary shared launcher's dataset lock as well.
                legacy=config.get('legacy_control_root') or str((Path(config['data_root'])/config.get('legacy_control_relative','_batch_pipeline')).resolve())
                if legacy and Path(legacy).resolve()!=Path(config['control_root']).resolve():
                    stack.enter_context(self.runtime.batch_lock(dict(config,control_root=legacy)))
                stack.enter_context(self.runtime.batch_lock(config))
            stack.enter_context(self.platform.inhibit_sleep())
            return stack
        except BaseException:stack.close();raise


class BatchLock:
    """One combined writer, with process-start identity and fail-closed ownership."""
    def __init__(self,batch,platform):self.path=batch/'state/combined.lock';self.platform=platform;self.token=uuid.uuid4().hex
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        owner=dict(host=self.platform.host_identity(),process=self.platform.process_identity(os.getpid()),token=self.token,started_utc=utc())
        if owner['process'] is None:raise RuntimeError('Cannot identify combined batch process')
        try:self.path.mkdir()
        except FileExistsError:
            p=self.path/'owner.json'
            if not p.is_file():raise BlockingIOError('An incomplete combined batch lock exists; no files were removed')
            previous=read(p)
            if previous.get('host')!=owner['host'] or self.platform.same_process(previous):raise BlockingIOError('This combined batch is already running or owned by another host')
            os.replace(self.path,self.path.with_name('combined.lock.stale_'+uuid.uuid4().hex));self.path.mkdir()
        atomic(self.path/'owner.json',owner);return self
    def __exit__(self,*args):
        p=self.path/'owner.json'
        if p.is_file() and read(p).get('token')==self.token:p.unlink();self.path.rmdir()


def stop_paths(batch,datasets):
    return [batch/'state/STOP_AFTER_CURRENT_CELL']+[Path(d['config']['control_root'])/'modular/state/STOP_AFTER_CURRENT_CELL' for d in datasets]

def request_stop(batch,datasets):
    for path in stop_paths(batch,datasets):atomic(path,dict(requested_utc=utc(),scope='Only the BR4/BR5 extension batch'))
    print('Stop requested: finish the current cell, then stop the entire combined batch.',flush=True)

def clear_stop(batch,datasets):
    for path in stop_paths(batch,datasets):
        if path.exists():os.replace(path,path.with_name(path.name+'.cleared_'+uuid.uuid4().hex))


def new_state(manifest,datasets,backend):
    return dict(schema_version=1,batch_id=manifest['batch_id'],status='PROGRESS',created_utc=utc(),
        cells={},summary_failures=[],old_cells_excluded_from_execution=manifest['existing_cells'],
        expected_new_cells=manifest['new_cells'],full_summary_cells=manifest['summary_cells'],
        families={d['molecule']:backend.family(d) for d in datasets},new_cells_started=0)

def state_progress(state):
    cells=state['cells'];state['completed_new_cells']=sum(r.get('status')=='complete' for r in cells.values())
    state['failed_new_cells']=sum(r.get('status') in ('failed','publication_incomplete') for r in cells.values())
    state['remaining_new_cells']=state['expected_new_cells']-state['completed_new_cells']
    state['updated_utc']=utc();return state


def execute_queue(manifest,datasets,backend,batch,action='start',save=None):
    """Control-flow unit, injectable for synthetic tests; only explicit new IDs run."""
    require_first=bool(manifest.get('require_first_complete_cell',False))
    gate_release=manifest.get('release_relative')
    if require_first and not isinstance(gate_release,str):
        raise ValueError('First-cell completion gate requires the pinned release_relative')
    if require_first and not gate_release.strip():
        raise ValueError('First-cell completion gate requires a non-empty release_relative')
    batch=Path(batch);path=batch/'state/batch_state.json'
    state=read(path) if path.is_file() else new_state(manifest,datasets,backend)
    families={d['molecule']:backend.family(d) for d in datasets}
    if state.get('batch_id')!=manifest['batch_id'] or state.get('families')!=families:
        if path.exists():os.replace(path,path.with_name('batch_state.previous_'+uuid.uuid4().hex+'.json'))
        state=new_state(manifest,datasets,backend)
    state.pop('stopped_after_cell',None)
    state.update(status='PROGRESS',started_utc=utc(),summary_failures=[])
    persist=save or (lambda value:atomic(path,state_progress(value)))
    stop=batch/'state/STOP_AFTER_CURRENT_CELL'
    if action in ('start','resume'):clear_stop(batch,datasets)
    persist(state)
    def first_cell_pending():
        return require_first and state.get('first_complete_cell_verified_release')!=gate_release
    def stop_first_cell(sid,reason):
        # Failed rows and committed stage checkpoints remain available for resume.
        # Do not set stopped_after_cell: this failure must produce a non-zero exit.
        state.pop('active_sample',None);state.pop('active_molecule',None)
        state.update(status='STOPPED_FIRST_CELL_FAILURE',last_error=reason,ended_utc=utc(),
            first_cell_gate_failure=dict(release_relative=gate_release,sample_id=sid,reason=reason,utc=utc()))
        state_progress(state);persist(state)
        print(sid+' | FIRST CELL FAILED: '+reason+'; remaining queue was not started. Resume retries committed checkpoints.',flush=True)
        return state
    def summarize(d,phase):
        failed=[sid for sid,r in state['cells'].items() if r.get('molecule')==d['molecule'] and r.get('status') in ('failed','publication_incomplete')]
        try:
            result=backend.summarize(d,failed);state.setdefault('summaries',{})[d['molecule']]=dict(phase=phase,result=result,updated_utc=utc())
            state['summary_failures']=[row for row in state['summary_failures'] if row['molecule']!=d['molecule']]
            if result.get('missing'):state.setdefault('summary_missing',{})[d['molecule']]=result['missing']
            else:state.setdefault('summary_missing',{}).pop(d['molecule'],None)
        except BaseException as error:
            if isinstance(error,(KeyboardInterrupt,SystemExit)) or backend.infrastructure(error):raise
            failure=dict(molecule=d['molecule'],phase=phase,error=str(error),traceback=traceback.format_exc(),utc=utc())
            state['summary_failures'].append(failure);state.setdefault('summary_failure_history',[]).append(failure)
    try:
        for d in datasets:backend.validate(d)
        for d in datasets:
            if stop.exists():break
            summarize(d,'startup')
            for sample in d['new_samples']:
                if stop.exists():break
                sid=sample.sample_id;stamp=None;previous=state['cells'].get(sid,{})
                try:stamp=backend.source_stamp(d,sample)
                except Exception as error:
                    if backend.infrastructure(error):raise
                    state['cells'][sid]=dict(molecule=d['molecule'],status='failed',family=families[d['molecule']],source_stamp=None,error=str(error),ended_utc=utc())
                    persist(state)
                    if first_cell_pending():return stop_first_cell(sid,'Source preparation failed: '+str(error))
                    continue
                if previous.get('status')=='complete' and previous.get('family')==families[d['molecule']] and previous.get('source_stamp')==stamp and backend.completion_outputs_valid(d,sample,previous):
                    print(sid+' | already complete; skipped',flush=True);continue
                state['active_sample']=sid;state['active_molecule']=d['molecule'];persist(state)
                verifying_first=first_cell_pending()
                if verifying_first:
                    print(sid+' | FIRST CELL: science and all cell figures must complete before the remaining queue starts.',flush=True)
                print(sid+' | extension cell ('+str(state_progress(state)['completed_new_cells'])+'/'+str(manifest['new_cells'])+' complete)',flush=True)
                try:
                    result=backend.run_cell(d,sample)
                    status=result.get('status','failed')
                    if status=='complete' and result.get('science_complete') is False:status='failed'
                    if verifying_first and status=='complete':
                        if result.get('science_complete') is not True:status='failed'
                        elif result.get('publication_complete') is not True:status='publication_incomplete'
                    row=dict(molecule=d['molecule'],status=status,family=families[d['molecule']],source_stamp=stamp,result=result,ended_utc=utc())
                    state['new_cells_started']+=int(result.get('scientific_stage_calls',0)>0)
                except BaseException as error:
                    if isinstance(error,(KeyboardInterrupt,SystemExit)) or backend.infrastructure(error):raise
                    row=dict(molecule=d['molecule'],status='failed',family=families[d['molecule']],source_stamp=stamp,error=str(error),traceback=traceback.format_exc(),ended_utc=utc())
                state['cells'][sid]=row;state.pop('active_sample',None);persist(state)
                if verifying_first:
                    if row['status']!='complete':
                        reason=row.get('error') or 'Cell did not finish science and publication (status='+str(row['status'])+')'
                        return stop_first_cell(sid,reason)
                    state['first_complete_cell_verified_release']=gate_release
                    state['first_complete_cell_verification']=dict(release_relative=gate_release,sample_id=sid,verified_utc=utc())
                    state.pop('first_cell_gate_failure',None)
                    persist(state)
                    print(sid+' | FIRST CELL COMPLETE: science and cell figures are saved; continuing the remaining queue automatically.',flush=True)
                summarize(d,'after '+sid);persist(state)
            if stop.exists():break
        if stop.exists():
            state.update(status='PROGRESS',stopped_after_cell=True,ended_utc=utc())
        else:
            # Final summaries always use every configured BR1-5 cell, including
            # explicit unavailable/failed new cells; no missing cell becomes zero.
            for d in datasets:summarize(d,'final')
            state_progress(state)
            state.update(status='COMPLETE' if state['completed_new_cells']==manifest['new_cells'] and not state['summary_failures'] and not any(state.get('summary_missing',{}).values()) else 'COMPLETE_WITH_FAILURES_OR_EXCLUSIONS',ended_utc=utc())
        persist(state);return state
    except BaseException as error:
        backend.terminate()
        state.update(status='PROGRESS' if isinstance(error,KeyboardInterrupt) else 'STOPPED_INFRASTRUCTURE',last_error=str(error),ended_utc=utc())
        persist(state);raise
    finally:backend.terminate()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path)
    p.add_argument('--action',choices=['start','resume','stop','plan','summarize','refresh'],default='start');args=p.parse_args(argv)
    manifest,common,batch=manifest_paths(args.manifest,require_ready=args.action!='stop')
    if not common.is_dir() or not os.path.ismount(str(common.parent)):raise OSError('The external data drive must be mounted; no output was written')
    early_environment(batch);release=activate_pinned(common,manifest);backend=SharedBackend(common,manifest)
    backend.runtime.platform.install_stop_handlers();datasets=backend.load_datasets()
    if args.action=='stop':request_stop(batch,datasets);return 0
    if args.action=='plan':
        print(json.dumps(dict(mode='read_only_queue_plan',batch_id=manifest['batch_id'],pinned_release=str(release),new_cells=manifest['new_cells'],existing_cells_excluded=manifest['existing_cells'],summary_cells=manifest['summary_cells'],datasets=[dict(molecule=d['molecule'],run_cells=len(d['new_samples']),summary_cells=len(d['all_samples']),new_sample_ids=[s.sample_id for s in d['new_samples']]) for d in datasets],new_cells_started=0,scientific_stage_calls=0),indent=2));return 0
    with BatchLock(batch,backend.platform),backend.lock_contexts(datasets):
        if args.action in ('summarize','refresh'):
            complete=True
            for d in datasets:
                try:
                    result=backend.summarize(d) if args.action=='summarize' else backend.refresh(d)
                    if result.get('missing') or result.get('complete') is False:complete=False
                    print(json.dumps(dict(molecule=d['molecule'],result=result),indent=2),flush=True)
                except Exception as error:
                    if backend.infrastructure(error):raise
                    complete=False
                    print(json.dumps(dict(molecule=d['molecule'],complete=False,error=str(error),scientific_upstream_calls=0),indent=2),flush=True)
            return 0 if complete else 1
        print('BR4/BR5 extension: only '+str(manifest['new_cells'])+' new cells; '+str(manifest['existing_cells'])+' old cells are summary-only. Five-replica summaries include '+str(manifest['summary_cells'])+' cells.',flush=True)
        state=execute_queue(manifest,datasets,backend,batch,args.action)
        print(json.dumps({k:state.get(k) for k in ('batch_id','status','completed_new_cells','failed_new_cells','remaining_new_cells','old_cells_excluded_from_execution')},indent=2),flush=True)
        return 0 if state['status']=='COMPLETE' or state.get('stopped_after_cell') else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:print('Interrupted; committed stage checkpoints are retained.',file=sys.stderr);raise SystemExit(130)
    except BlockingIOError as error:print('BATCH BUSY: '+str(error),file=sys.stderr);raise SystemExit(75)
    except Exception:traceback.print_exc();raise SystemExit(1)
