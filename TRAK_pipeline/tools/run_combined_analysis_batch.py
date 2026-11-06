#!/usr/bin/env python3
"""One Windows entry point for pending BR4/BR5 cells, then HaloTag-only controls.

Each child retains its own queue state, completion reports, locks and summaries.
This controller contains no scientific algorithms and never pools controls with
TRAK targets. A global stop survives the transition between the two queues.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
import uuid


def sibling(name):
    path=Path(__file__).with_name(name+'.py')
    spec=importlib.util.spec_from_file_location('_combined_'+name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


extension=sibling('run_replica_extension_batch')


def manifest_paths(path,require_ready=True):
    path=Path(path).resolve();m=extension.read(path);batch=path.parent
    common=(batch/m.get('common_root_relative','../..')).resolve()
    if m.get('schema_version')!=1 or m.get('path_basis')!='common_root':raise ValueError('Unsupported combined manifest')
    if not batch.is_relative_to(common/'batches'):raise ValueError('Combined manifest escapes the shared batches directory')
    if require_ready and (not m.get('ready_for_execution') or not m.get('release_relative')):raise ValueError('Combined batch is not frozen and ready')
    jobs=m.get('jobs',[])
    if not jobs or len({j['job_id'] for j in jobs})!=len(jobs):raise ValueError('Combined job identities must be unique')
    seen=set()
    for job in jobs:
        if job.get('kind') not in ('replica_extension','control'):raise ValueError('Unknown child queue type')
        child=(common/job['manifest_relative']).resolve()
        if not child.is_relative_to(common/'batches') or not child.is_file() or child==path:raise ValueError('Missing, recursive or out-of-package child manifest')
        child_manifest=extension.read(child)
        if child_manifest.get('release_relative')!=m.get('release_relative'):raise ValueError('All child queues must use the combined pinned release')
        if require_ready and not child_manifest.get('ready_for_execution'):raise ValueError('A child queue is not ready')
        for d in child_manifest.get('datasets',[]):
            for sid in d['new_sample_ids']:
                if sid in seen:raise ValueError('Sample ID is queued by more than one child job: '+sid)
                seen.add(sid)
    return m,common,batch


def child_stop_paths(common,manifest):
    """Resolve only declared control paths; does not load runtimes or sample data."""
    paths=[]
    for job in manifest['jobs']:
        child=(common/job['manifest_relative']).resolve();cm=extension.read(child)
        paths.append(child.parent/'state/STOP_AFTER_CURRENT_CELL')
        for record in cm.get('datasets',[]):
            config_path=(common/record['config_relative']).resolve()
            if not config_path.is_relative_to(common):raise ValueError('Child configuration escapes shared package')
            config=extension.read(config_path);root=Path(config['data_root'])
            root=(root if root.is_absolute() else config_path.parent/root).resolve()
            control=(root/config.get('control_relative','_batch_pipeline')).resolve()
            if not root.is_relative_to(common.parent) or root==common.parent or not control.is_relative_to(root):raise ValueError('Control writes must remain inside a declared dataset on the external drive')
            paths.append(control/'modular/state/STOP_AFTER_CURRENT_CELL')
    return list(dict.fromkeys(paths))


def request_stop(common,manifest,batch):
    # Parent marker is committed first; a child started during the transition
    # checks it before clearing its ordinary resume markers.
    paths=[Path(batch)/'state/STOP_AFTER_CURRENT_CELL']+child_stop_paths(common,manifest)
    for path in paths:extension.atomic(path,dict(requested_utc=extension.utc(),scope='All pending TRAK and HaloTag-only queues'))
    print('Stop requested: finish the active cell, then stop both queues.',flush=True)


def install_external_stop_guard(module):
    external=os.environ.get('TRAK_COMBINED_STOP_FILE')
    if not external:return
    original=module.clear_stop
    def guarded_clear(batch,datasets):
        if Path(external).exists():module.request_stop(batch,datasets)
        else:original(batch,datasets)
    module.clear_stop=guarded_clear


def collect_plan(manifest,common,batch):
    """Inspect exact queues and valid completed records without registering or running."""
    plans=[]
    for job in manifest['jobs']:
        child_path=common/job['manifest_relative'];cm=extension.read(child_path)
        backend=extension.SharedBackend(common,cm) if job['kind']=='replica_extension' else sibling('run_control_batch').ControlBackend(common,cm)
        datasets=backend.load_datasets();state_path=child_path.parent/'state/batch_state.json'
        state=extension.read(state_path) if state_path.is_file() else extension.new_state(cm,datasets,backend)
        completed=[]
        for d in datasets:
            family=backend.family(d)
            for sample in d['new_samples']:
                row=state.get('cells',{}).get(sample.sample_id,{})
                valid=False
                if row.get('status')=='complete' and row.get('family')==family:
                    valid=row.get('source_stamp')==backend.source_stamp(d,sample) and backend.completion_outputs_valid(d,sample,row)
                if not valid and job['kind']=='control':valid=sibling('run_control_batch').reusable_completion(d,sample,backend) is not None
                if valid:completed.append(sample.sample_id)
        plans.append(dict(job_id=job['job_id'],kind=job['kind'],manifest_relative=job['manifest_relative'],
            queued_cells=cm['new_cells'],completed_cells=len(completed),pending_cells=cm['new_cells']-len(completed),
            completed_sample_ids=completed,summary_cells=cm['summary_cells'],summary_scope='Independent '+', '.join(d['molecule'] for d in datasets)))
    return dict(mode='read_only_queue_plan',batch_id=manifest['batch_id'],jobs=plans,
        queued_cells=sum(p['queued_cells'] for p in plans),completed_cells=sum(p['completed_cells'] for p in plans),
        pending_cells=sum(p['pending_cells'] for p in plans),scientific_stage_calls=0,queue_state_writes=0)


def child_command(release,job,common,action):
    if job['kind']=='replica_extension':
        # Wrapper changes only stop handling in memory. The original controller
        # receives its original manifest and state path without translation.
        return [sys.executable,'-B','-s','-u',str(release/'tools/run_combined_analysis_batch.py'),
            '--child-kind','replica_extension','--manifest',str(common/job['manifest_relative']),'--action',action]
    return [sys.executable,'-B','-s','-u',str(release/'tools/run_control_batch.py'),
        '--manifest',str(common/job['manifest_relative']),'--action',action]


def run_child_process(common,manifest,batch,release,platform,job,action):
    command=child_command(release,job,common,action)
    logs=Path(batch)/'logs';logs.mkdir(parents=True,exist_ok=True)
    log_path=logs/(job['job_id']+'_'+extension.utc().replace(':','').replace('+','_')+'.log')
    env=dict(os.environ,TRAK_COMBINED_STOP_FILE=str(Path(batch)/'state/STOP_AFTER_CURRENT_CELL'))
    print('COMBINED_JOB '+job['job_id']+' | '+action,flush=True)
    print('COMMAND_QUEUE '+json.dumps(command),flush=True)
    process=None
    try:
        with log_path.open('w',encoding='utf-8') as log:
            # Uses production Windows Job Objects and Mac process groups. Only
            # this controller's owned child tree is terminated on Ctrl+C.
            process=platform.spawn_stage(command,release,log,env)
            forwarded=False
            with log_path.open(encoding='utf-8',errors='replace') as reader:
                while process.poll() is None:
                    for line in reader:print(line,end='',flush=True)
                    if (Path(batch)/'state/STOP_AFTER_CURRENT_CELL').exists() and not forwarded:
                        request_stop(common,manifest,batch);forwarded=True
                    time.sleep(.2)
                for line in reader:print(line,end='',flush=True)
            return process.wait()
    except BaseException:
        if process is not None:platform.terminate_tree(process)
        raise
    finally:
        if process is not None:
            close=getattr(process,'close',None)
            if close:close()


def execute_jobs(manifest,common,batch,run_job,action='start',save=None):
    """Injectable queue sequencing; this function never calls scientific stages."""
    batch=Path(batch);state_path=batch/'state/batch_state.json';stop=batch/'state/STOP_AFTER_CURRENT_CELL'
    state=extension.read(state_path) if state_path.is_file() else dict(schema_version=1,batch_id=manifest['batch_id'],jobs={})
    if state.get('batch_id')!=manifest['batch_id']:raise ValueError('Combined state belongs to a different batch; preserved')
    state.update(status='PROGRESS',action=action,started_utc=extension.utc());state.pop('stopped_after_cell',None)
    persist=save or (lambda value:extension.atomic(state_path,value))
    if action in ('start','resume') and stop.exists():os.replace(stop,stop.with_name(stop.name+'.cleared_'+uuid.uuid4().hex))
    persist(state);partial=False
    try:
        for job in manifest['jobs']:
            if stop.exists():state.update(stopped_after_cell=True);break
            child_path=common/job['manifest_relative'];child_state_path=child_path.parent/'state/batch_state.json'
            before=extension.read(child_state_path) if child_state_path.is_file() else None
            state['active_job']=job['job_id'];persist(state)
            code=run_job(job,action)
            after=extension.read(child_state_path) if child_state_path.is_file() else {}
            status=after.get('status');changed=before!=after
            state['jobs'][job['job_id']]=dict(exit_code=code,child_status=status,ended_utc=extension.utc(),
                child_manifest_relative=job['manifest_relative'],completed_new_cells=after.get('completed_new_cells'),remaining_new_cells=after.get('remaining_new_cells'))
            state.pop('active_job',None);persist(state)
            if code in (130,74,75):raise RuntimeError('Child queue stopped with interruption, infrastructure or duplicate-writer status '+str(code))
            if action in ('summarize','refresh'):
                if code:partial=True
                continue
            if stop.exists() or after.get('stopped_after_cell'):
                state.update(stopped_after_cell=True);break
            if status=='COMPLETE' and code==0:continue
            if status=='COMPLETE_WITH_FAILURES_OR_EXCLUSIONS' and (code==0 or (code==1 and changed)):
                partial=True;continue
            raise RuntimeError('Child queue did not finish safely: '+job['job_id']+' (exit='+str(code)+', status='+str(status)+'); the next queue was not started')
        state.update(status='PROGRESS' if state.get('stopped_after_cell') else ('COMPLETE_WITH_FAILURES_OR_EXCLUSIONS' if partial else 'COMPLETE'),ended_utc=extension.utc())
        persist(state);return state
    except BaseException as error:
        state.update(status='PROGRESS' if isinstance(error,KeyboardInterrupt) else 'STOPPED_INFRASTRUCTURE',last_error=str(error),ended_utc=extension.utc())
        persist(state);raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--action',choices=['start','resume','stop','plan','summarize','refresh'],default='start')
    parser.add_argument('--child-kind',choices=['replica_extension'],help=argparse.SUPPRESS);args=parser.parse_args(argv)
    if args.child_kind:
        install_external_stop_guard(extension)
        return extension.main(['--manifest',str(args.manifest),'--action',args.action])
    manifest,common,batch=manifest_paths(args.manifest,require_ready=args.action not in ('stop','plan'))
    if not common.is_dir() or not os.path.ismount(str(common.parent)):raise OSError('External data drive is not mounted')
    if args.action=='stop':request_stop(common,manifest,batch);return 0
    extension.early_environment(batch);release=extension.activate_pinned(common,manifest)
    from trak_pipeline.runtime import platform
    platform.install_stop_handlers()
    if args.action=='plan':print(json.dumps(collect_plan(manifest,common,batch),indent=2),flush=True);return 0
    with extension.BatchLock(batch,platform),platform.inhibit_sleep():
        state=execute_jobs(manifest,common,batch,lambda job,action:run_child_process(common,manifest,batch,release,platform,job,action),args.action)
        print(json.dumps(state,indent=2),flush=True)
        return 0 if state['status']=='COMPLETE' or state.get('stopped_after_cell') else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:print('Interrupted; committed checkpoints are retained in both original queues.',file=sys.stderr);raise SystemExit(130)
    except BlockingIOError as error:print('BATCH BUSY: '+str(error),file=sys.stderr);raise SystemExit(75)
    except Exception:traceback.print_exc();raise SystemExit(1)
