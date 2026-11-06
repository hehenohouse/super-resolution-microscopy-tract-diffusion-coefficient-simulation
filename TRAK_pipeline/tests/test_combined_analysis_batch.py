"""Bounded controller tests: no detection, fitting, segmentation or TARDIS calls."""
from __future__ import annotations
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

def load(name):
    spec=importlib.util.spec_from_file_location('test_'+name,ROOT/'tools'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

outer=load('run_combined_analysis_batch');control=load('run_control_batch')


class Fixture(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'validation/combined_analysis_batch_20261104';parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='unit_',dir=parent);self.addCleanup(self.temp.cleanup)
        self.common=Path(self.temp.name);self.batch=self.common/'batches/combined'
        self.manifest=dict(schema_version=1,batch_id='combined',jobs=[
            dict(job_id='TRAK',kind='replica_extension',manifest_relative='batches/TRAK/manifest.json'),
            dict(job_id='Halo',kind='control',manifest_relative='batches/Halo/manifest.json')])
    def write_child_state(self,job,**values):
        p=(self.common/job['manifest_relative']).parent/'state/batch_state.json'
        outer.extension.atomic(p,values);return p
    def run_jobs(self,fn,action='start'):
        return outer.execute_jobs(self.manifest,self.common,self.batch,fn,action)


class CombinedQueueTests(Fixture):
    def test_success_preserves_child_state_locations_and_order(self):
        seen=[]
        def run(job,action):
            seen.append((job['job_id'],action,job['manifest_relative']))
            self.write_child_state(job,status='COMPLETE',completed_new_cells=3)
            return 0
        result=self.run_jobs(run)
        self.assertEqual(result['status'],'COMPLETE');self.assertEqual([r[0] for r in seen],['TRAK','Halo'])
        self.assertEqual(seen[0][2],'batches/TRAK/manifest.json')
    def test_regular_cell_failures_continue_control_queue(self):
        seen=[]
        def run(job,action):
            seen.append(job['job_id'])
            self.write_child_state(job,status='COMPLETE_WITH_FAILURES_OR_EXCLUSIONS' if job['job_id']=='TRAK' else 'COMPLETE')
            return 1 if job['job_id']=='TRAK' else 0
        result=self.run_jobs(run)
        self.assertEqual(seen,['TRAK','Halo']);self.assertEqual(result['status'],'COMPLETE_WITH_FAILURES_OR_EXCLUSIONS')
    def test_infrastructure_and_first_gate_failures_never_start_control(self):
        for status in ('STOPPED_INFRASTRUCTURE','STOPPED_FIRST_CELL_FAILURE','PROGRESS',None):
            with self.subTest(status=status):
                seen=[]
                def run(job,action):
                    seen.append(job['job_id']);self.write_child_state(job,status=status);return 1
                with self.assertRaises(RuntimeError):self.run_jobs(run)
                self.assertEqual(seen,['TRAK'])
    def test_stale_failure_terminal_state_does_not_hide_early_traceback(self):
        for job in self.manifest['jobs']:self.write_child_state(job,status='COMPLETE_WITH_FAILURES_OR_EXCLUSIONS')
        seen=[]
        def run(job,action):seen.append(job['job_id']);return 1
        with self.assertRaises(RuntimeError):self.run_jobs(run)
        self.assertEqual(seen,['TRAK'])
    def test_global_stop_after_active_queue_prevents_control_start(self):
        seen=[]
        def run(job,action):
            seen.append(job['job_id']);self.write_child_state(job,status='PROGRESS',stopped_after_cell=True)
            outer.extension.atomic(self.batch/'state/STOP_AFTER_CURRENT_CELL',{})
            return 0
        result=self.run_jobs(run)
        self.assertEqual(seen,['TRAK']);self.assertTrue(result['stopped_after_cell']);self.assertEqual(result['status'],'PROGRESS')
    def test_interrupt_preserves_child_checkpoint_and_stops_sequence(self):
        job=self.manifest['jobs'][0];checkpoint=self.write_child_state(job,status='PROGRESS',completed_new_cells=3)
        before=checkpoint.read_bytes()
        with self.assertRaises(KeyboardInterrupt):self.run_jobs(lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertEqual(checkpoint.read_bytes(),before)
        self.assertEqual(outer.extension.read(self.batch/'state/batch_state.json')['status'],'PROGRESS')
    def test_child_command_uses_original_manifest_and_owned_wrapper(self):
        command=outer.child_command(self.common/'release',self.manifest['jobs'][0],self.common,'resume')
        self.assertIn('--child-kind',command);self.assertIn(str(self.common/'batches/TRAK/manifest.json'),command)
        self.assertIn('resume',command)
    def test_parent_stop_cannot_be_cleared_in_child_start_race(self):
        marker=self.batch/'state/STOP_AFTER_CURRENT_CELL';outer.extension.atomic(marker,{})
        calls=[];module=types.SimpleNamespace(clear_stop=lambda *a:calls.append('cleared'),request_stop=lambda *a:calls.append('stopped'))
        with patch.dict(os.environ,TRAK_COMBINED_STOP_FILE=str(marker)):
            outer.install_external_stop_guard(module);module.clear_stop(self.batch,[])
        self.assertEqual(calls,['stopped']);self.assertTrue(marker.is_file())
    def test_no_stop_request_keeps_normal_resume_behavior(self):
        calls=[];module=types.SimpleNamespace(clear_stop=lambda *a:calls.append('cleared'),request_stop=lambda *a:calls.append('stopped'))
        with patch.dict(os.environ,TRAK_COMBINED_STOP_FILE=str(self.batch/'missing')):
            outer.install_external_stop_guard(module);module.clear_stop(self.batch,[])
        self.assertEqual(calls,['cleared'])
    def test_owned_process_tree_terminated_on_interrupt(self):
        calls=[]
        class Process:
            def poll(self):raise KeyboardInterrupt()
            def close(self):calls.append('closed')
        process=Process()
        platform=types.SimpleNamespace(spawn_stage=lambda *a:process,terminate_tree=lambda p:calls.append('terminated'))
        with self.assertRaises(KeyboardInterrupt):
            outer.run_child_process(self.common,self.manifest,self.batch,self.common/'release',platform,self.manifest['jobs'][0],'resume')
        self.assertEqual(calls,['terminated','closed'])
    def test_duplicate_writer_lock_is_rejected(self):
        platform=types.SimpleNamespace(host_identity=lambda:'test-host',process_identity=lambda pid:dict(pid=pid),same_process=lambda owner:True)
        with outer.extension.BatchLock(self.batch,platform):
            with self.assertRaises(BlockingIOError):
                with outer.extension.BatchLock(self.batch,platform):pass
    def test_read_only_plan_counts_3_plus_1_completed_and_172_pending(self):
        records=[];datasets={}
        for job,n in zip(self.manifest['jobs'],(156,20)):
            molecule='TRAK1' if n==156 else 'HaloTag_only';samples=[types.SimpleNamespace(sample_id=molecule+'_'+str(i)) for i in range(n)]
            d=dict(molecule=molecule,new_samples=samples,all_samples=samples,config={});datasets[molecule]=[d]
            cm=dict(batch_id=job['job_id'],new_cells=n,existing_cells=211 if n==156 else 0,summary_cells=367 if n==156 else 20,datasets=[dict(molecule=molecule)])
            outer.extension.atomic(self.common/job['manifest_relative'],cm)
            if n==156:self.write_child_state(job,cells={s.sample_id:dict(status='complete',family='f',source_stamp={'C1':'a'}) for s in samples[:3]})
        class Backend:
            def __init__(self,common,manifest):self.manifest=manifest
            def load_datasets(self):return datasets[self.manifest['datasets'][0]['molecule']]
            def family(self,d):return 'f'
            def source_stamp(self,d,s):return {'C1':'a'}
            def completion_outputs_valid(self,*args):return True
        fake=types.SimpleNamespace(ControlBackend=Backend,reusable_completion=lambda d,s,b:dict(valid=True) if s.sample_id=='HaloTag_only_0' else None)
        before={p:p.read_bytes() for p in self.common.rglob('*.json')}
        with patch.object(outer.extension,'SharedBackend',Backend),patch.object(outer,'sibling',return_value=fake):
            plan=outer.collect_plan(self.manifest,self.common,self.batch)
        self.assertEqual((plan['completed_cells'],plan['pending_cells']),(4,172))
        self.assertEqual(plan['scientific_stage_calls'],0);self.assertEqual(plan['queue_state_writes'],0)
        self.assertEqual({p:p.read_bytes() for p in self.common.rglob('*.json')},before)


class CompletionAdoptionTests(Fixture):
    def setUp(self):
        super().setUp();self.samples=[types.SimpleNamespace(sample_id='Halo_T1'),types.SimpleNamespace(sample_id='Halo_T2')]
        self.d=dict(molecule='HaloTag_only',config=dict(data_root=str(self.common),control_root=str(self.common/'pilot')),new_samples=self.samples)
        self.cm=dict(batch_id='Halo',new_cells=2,existing_cells=0,summary_cells=2,release_relative='release')
        self.stamp={'C1':'c1','C2':'c2','roi':'roi'};self.calls=[]
        self.backend=types.SimpleNamespace(family=lambda d:'f',source_stamp=lambda d,s:dict(self.stamp),completion_outputs_valid=lambda *a:True)
        self.path=control.completion_report_path(self.d,self.samples[0],'f')
        control.extension.atomic(self.path,dict(sample_id='Halo_T1',method_family='f',science_complete=True,publication_complete=True,
            artifacts={k:dict(id=v) for k,v in self.stamp.items()},completed_utc='earlier'))
    def test_adopts_pilot_without_touching_original_completion_or_running_cell(self):
        before=self.path.read_bytes();state,adopted=control.adopt_completed(self.cm,[self.d],self.backend,self.batch)
        self.assertEqual(adopted,['Halo_T1']);self.assertEqual(state['completed_new_cells'],1)
        self.assertEqual(state['cells']['Halo_T1']['result']['scientific_stage_calls'],0)
        self.assertEqual(before,self.path.read_bytes());self.assertFalse('run_cell' in self.backend.__dict__)
    def test_source_changes_or_missing_source_metadata_prevent_adoption(self):
        self.stamp['C1']='changed';self.assertIsNone(control.reusable_completion(self.d,self.samples[0],self.backend))
        self.stamp['C1']='c1';report=control.extension.read(self.path);del report['artifacts']['roi'];control.extension.atomic(self.path,report)
        self.assertIsNone(control.reusable_completion(self.d,self.samples[0],self.backend))
    def test_incomplete_outputs_cannot_be_adopted(self):
        self.backend.completion_outputs_valid=lambda *a:False
        self.assertIsNone(control.reusable_completion(self.d,self.samples[0],self.backend))
    def test_read_only_adoption_plan_does_not_write_state(self):
        state,adopted=control.adopt_completed(self.cm,[self.d],self.backend,self.batch,write=False)
        self.assertEqual(adopted,['Halo_T1']);self.assertEqual(state['remaining_new_cells'],1)
        self.assertFalse((self.batch/'state/batch_state.json').exists())
    def test_adopted_completed_cell_is_skipped_by_production_queue(self):
        control.adopt_completed(self.cm,[self.d],self.backend,self.batch)
        def run(d,s):self.calls.append(s.sample_id);return dict(status='complete',science_complete=True,publication_complete=True,scientific_stage_calls=1)
        self.backend.run_cell=run;self.backend.validate=lambda d:None;self.backend.summarize=lambda *a:{}
        self.backend.infrastructure=lambda e:False;self.backend.terminate=lambda:None
        state=control.extension.execute_queue(self.cm,[self.d],self.backend,self.batch,'resume')
        self.assertEqual(self.calls,['Halo_T2']);self.assertEqual(state['status'],'COMPLETE')


if __name__=='__main__':unittest.main(verbosity=2)
