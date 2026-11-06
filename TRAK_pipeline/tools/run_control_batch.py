#!/usr/bin/env python3
"""Resume an isolated negative-control queue using the production stage runner.

No detection, fitting, mask or TARDIS algorithms are implemented here. Existing
completed reports are registered only after source and committed-output checks.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def load_extension():
    path=Path(__file__).with_name('run_replica_extension_batch.py')
    spec=importlib.util.spec_from_file_location('_control_extension',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


extension=load_extension()


def manifest_paths(path,require_ready=True):
    path=Path(path).resolve();m=extension.read(path);batch=path.parent
    common=(batch/m.get('common_root_relative','../..')).resolve()
    if m.get('schema_version')!=1 or m.get('path_basis')!='common_root':raise ValueError('Unsupported control manifest')
    if not batch.is_relative_to(common/'batches'):raise ValueError('Control manifest escapes the shared batches directory')
    if require_ready and (not m.get('ready_for_execution') or not m.get('release_relative')):raise ValueError('Control batch is not frozen and ready')
    if m.get('experiment_type')!='negative_control' or m.get('merge_with_target_datasets') is not False:raise ValueError('Negative-control membership must be explicit and independent')
    records=m.get('datasets',[])
    if len(records)!=1 or records[0].get('molecule')!='HaloTag_only':raise ValueError('This queue requires the isolated HaloTag-only dataset')
    d=records[0];ids=d['new_sample_ids']
    if len(ids)!=len(set(ids)) or len(ids)!=d['new_cells'] or len(ids)!=m['new_cells']:raise ValueError('Invalid control queue IDs/count')
    config=(common/d['config_relative']).resolve()
    if not config.is_relative_to(common) or not config.is_file():raise ValueError('Missing or out-of-package control configuration')
    if m.get('require_first_complete_cell'):raise ValueError('Control batch must not introduce a pilot qualification gate')
    return m,common,batch


class ControlBackend(extension.SharedBackend):
    def load_datasets(self):
        values=[]
        for record in self.manifest['datasets']:
            config=self.api.load_config(self.common/record['config_relative'])
            samples=self.api.discover_samples(config);byid={s.sample_id:s for s in samples}
            if len(samples)!=record['total_cells'] or len(samples)!=len(byid):raise ValueError('Control inventory identity/count differs')
            if config.get('molecule')!='HaloTag_only':raise ValueError('A target-protein dataset cannot enter the control queue')
            if any(s.metadata.get('molecule')!='HaloTag_only' for s in samples):raise ValueError('Mixed target/control membership')
            if set(self.manifest.get('conditions',[]))!={s.metadata.get('genotype') for s in samples}:raise ValueError('Control conditions differ from the explicit manifest')
            if set(self.manifest.get('replicas',[]))!={s.metadata.get('BR') for s in samples}:raise ValueError('Control replicas differ from the explicit manifest')
            new=[]
            for sid in record['new_sample_ids']:
                if sid not in byid:raise ValueError('Control queue ID is outside the declared inventory: '+sid)
                new.append(byid[sid])
            values.append(dict(record=record,config=config,all_samples=samples,new_samples=new,molecule=record['molecule']))
        self.datasets=values;return values


def completion_report_path(d,s,family):
    return Path(d['config']['control_root'])/'modular/state/cells'/s.sample_id/(family+'.json')


def reusable_completion(d,s,backend):
    """Read-only qualification of one existing report; never calls run_cell."""
    family=backend.family(d);path=completion_report_path(d,s,family)
    if not path.is_file():return None
    try:
        report=extension.read(path)
        if report.get('sample_id')!=s.sample_id or report.get('method_family')!=family:return None
        if report.get('science_complete') is not True or report.get('publication_complete') is not True:return None
        stamp=backend.source_stamp(d,s)
        # Sources are persisted in the completed report. Never substitute newly
        # computed fingerprints for missing or mismatching provenance.
        saved={k:report['artifacts'][k]['id'] for k in stamp}
        if not {'C1','C2','roi'}.issubset(stamp) or saved!=stamp:return None
        previous=dict(molecule=d['molecule'],status='complete',family=family,source_stamp=stamp)
        if not backend.completion_outputs_valid(d,s,previous):return None
        previous.update(result=dict(sample_id=s.sample_id,status='complete',science_complete=True,publication_complete=True,scientific_stage_calls=0),
            adopted_existing_completion=True,completion_report_relative=str(path.relative_to(Path(d['config']['data_root']))),
            original_completed_utc=report.get('completed_utc'),ended_utc=extension.utc())
        return previous
    except (OSError,ValueError,KeyError,TypeError):return None


def adopt_completed(manifest,datasets,backend,batch,write=True):
    path=Path(batch)/'state/batch_state.json'
    state=extension.read(path) if path.is_file() else extension.new_state(manifest,datasets,backend)
    families={d['molecule']:backend.family(d) for d in datasets}
    if state.get('batch_id')!=manifest['batch_id'] or state.get('families')!=families:
        if path.is_file():raise ValueError('Control queue state belongs to a different batch or method family; preserved without modification')
        state=extension.new_state(manifest,datasets,backend)
    adopted=[]
    for d in datasets:
        for sample in d['new_samples']:
            previous=state['cells'].get(sample.sample_id,{})
            if previous.get('status')=='complete':continue
            row=reusable_completion(d,sample,backend)
            if row is not None:state['cells'][sample.sample_id]=row;adopted.append(sample.sample_id)
    extension.state_progress(state)
    if write and adopted:extension.atomic(path,state)
    return state,adopted


def install_external_stop_guard(module=extension):
    external=os.environ.get('TRAK_COMBINED_STOP_FILE')
    if not external:return
    old=module.clear_stop
    def guarded_clear(batch,datasets):
        if Path(external).exists():module.request_stop(batch,datasets)
        else:old(batch,datasets)
    module.clear_stop=guarded_clear


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--action',choices=['start','resume','stop','plan','summarize','refresh'],default='start');args=parser.parse_args(argv)
    manifest,common,batch=manifest_paths(args.manifest,require_ready=args.action not in ('stop','plan'))
    if not common.is_dir() or not os.path.ismount(str(common.parent)):raise OSError('External data drive is not mounted')
    extension.early_environment(batch);release=extension.activate_pinned(common,manifest)
    backend=ControlBackend(common,manifest);backend.platform.install_stop_handlers();datasets=backend.load_datasets()
    install_external_stop_guard()
    if args.action=='stop':extension.request_stop(batch,datasets);return 0
    if args.action=='plan':
        state,adopted=adopt_completed(manifest,datasets,backend,batch,write=False)
        print(json.dumps(dict(mode='read_only_queue_plan',batch_id=manifest['batch_id'],pinned_release=str(release),
            queued_cells=manifest['new_cells'],completed_cells=state['completed_new_cells'],pending_cells=state['remaining_new_cells'],
            reusable_existing_reports=adopted,summary_cells=manifest['summary_cells'],scientific_stage_calls=0),indent=2));return 0
    with extension.BatchLock(batch,backend.platform),backend.lock_contexts(datasets):
        if args.action in ('summarize','refresh'):
            complete=True
            for d in datasets:
                try:
                    result=backend.summarize(d) if args.action=='summarize' else backend.refresh(d)
                    if result.get('missing') or result.get('complete') is False:complete=False
                    print(json.dumps(dict(molecule=d['molecule'],result=result),indent=2),flush=True)
                except Exception as error:
                    if backend.infrastructure(error):raise
                    complete=False;print(json.dumps(dict(molecule=d['molecule'],error=str(error),scientific_upstream_calls=0)),flush=True)
            return 0 if complete else 1
        state,adopted=adopt_completed(manifest,datasets,backend,batch)
        for sid in adopted:print(sid+' | existing complete pilot/result registered; no cell science rerun',flush=True)
        print('HaloTag-only: '+str(state['remaining_new_cells'])+' pending cells; independent WT/two-replica summaries.',flush=True)
        state=extension.execute_queue(manifest,datasets,backend,batch,args.action)
        print(json.dumps({k:state.get(k) for k in ('batch_id','status','completed_new_cells','failed_new_cells','remaining_new_cells')},indent=2),flush=True)
        return 0 if state['status']=='COMPLETE' or state.get('stopped_after_cell') else 1


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:print('Interrupted; committed checkpoints are retained.',file=sys.stderr);raise SystemExit(130)
    except BlockingIOError as error:print('BATCH BUSY: '+str(error),file=sys.stderr);raise SystemExit(75)
    except Exception:traceback.print_exc();raise SystemExit(1)
