"""Synthetic orchestration tests: never import or start scientific workers."""
from pathlib import Path
import importlib.util
import json
import tempfile
import types
import unittest

ROOT=Path(__file__).resolve().parents[1]
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
runner=load('_extension_controller','run_replica_extension_batch.py')
installer=load('_extension_install','install_replica_extension_launchers.py')

class InfrastructureError(RuntimeError):pass
class FakePlatform:
    @staticmethod
    def host_identity():return 'synthetic-host'
    @staticmethod
    def process_identity(pid):return dict(pid=pid,started='synthetic-start')
    @staticmethod
    def same_process(record):return True

class FakeBackend:
    def __init__(self,batch):self.batch=batch;self.calls=[];self.summaries=[];self.failed=set();self.infrastructure_ids=set();self.stop_on=None;self.terminated=0;self.version='same-science';self.stamps={};self.missing_outputs=set();self.cached_science=set();self.science_calls=[];self.interrupt_ids=set();self.summary_fail_once=False;self.missing_summary=False
    def family(self,d):return self.version
    def source_stamp(self,d,s):return self.stamps.get(s.sample_id,{'source':'same-saved-input'})
    def completion_outputs_valid(self,d,s,previous):return s.sample_id not in self.missing_outputs
    def validate(self,d):pass
    def infrastructure(self,e):return isinstance(e,InfrastructureError)
    def run_cell(self,d,s):
        self.calls.append(s.sample_id)
        if s.sample_id in self.interrupt_ids:raise KeyboardInterrupt('Synthetic Control-C')
        if s.sample_id in self.infrastructure_ids:raise InfrastructureError('Synthetic drive/software failure')
        if s.sample_id in self.failed:raise ValueError('Synthetic individual-cell failure')
        if s.sample_id==self.stop_on:runner.atomic(self.batch/'state/STOP_AFTER_CURRENT_CELL',{'requested':True})
        science=int(s.sample_id not in self.cached_science)
        if science:self.science_calls.append(s.sample_id)
        self.cached_science.add(s.sample_id);self.missing_outputs.discard(s.sample_id)
        return dict(status='complete',science_complete=True,publication_complete=True,scientific_stage_calls=science)
    def summarize(self,d,failed_ids=()):
        if self.summary_fail_once:
            self.summary_fail_once=False;raise ValueError('Synthetic transient summary publication failure')
        self.summaries.append((d['molecule'],tuple(failed_ids)));return dict(complete=True,summary_cells=len(d['all_samples']),missing=[{'sample_id':'old_missing','stage':'decomposition'}] if self.missing_summary and d['molecule']=='TRAK1' else [])
    def terminate(self):self.terminated+=1

def fixture(batch):
    def sample(sid,br):return types.SimpleNamespace(sample_id=sid,metadata={'BR':br})
    values=[]
    for protein,ids in [('TRAK1',['A4','A5']),('TRAK2',['B4'])]:
        new=[sample(sid,'BR4' if '4' in sid else 'BR5') for sid in ids];old=sample(protein+'_old','BR1')
        values.append(dict(molecule=protein,new_samples=new,all_samples=[old,*new],config=dict(control_root=str(batch/protein/'isolated_control'))))
    manifest=dict(batch_id='20261102_BR4_BR5',new_cells=3,existing_cells=2,summary_cells=5)
    return manifest,values

class ExtensionControlTests(unittest.TestCase):
    def test_only_new_cells_run_and_both_full_inventories_are_summarized(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=fixture(batch);backend=FakeBackend(batch)
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A4','A5','B4']);self.assertEqual(state['status'],'COMPLETE')
            self.assertEqual(state['old_cells_excluded_from_execution'],2)
            self.assertEqual([p for p,_ in backend.summaries[-2:]],['TRAK1','TRAK2'])
            self.assertEqual(state['completed_new_cells'],3)
    def test_stop_ends_whole_batch_and_start_resumes_without_repeating_completed(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=fixture(batch);backend=FakeBackend(batch);backend.stop_on='A4'
            oldstop=batch/'old_control/STOP_AFTER_CURRENT_CELL';oldstop.parent.mkdir();oldstop.write_text('old pause')
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A4']);self.assertTrue(state['stopped_after_cell']);self.assertEqual(state['status'],'PROGRESS')
            backend.stop_on=None;state=runner.execute_queue(manifest,data,backend,batch,'start')
            self.assertEqual(backend.calls,['A4','A5','B4']);self.assertEqual(state['status'],'COMPLETE')
            self.assertNotIn('stopped_after_cell',state);self.assertEqual(oldstop.read_text(),'old pause')
    def test_individual_failure_continues_and_final_summaries_include_failure(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=fixture(batch);backend=FakeBackend(batch);backend.failed={'A4'}
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A4','A5','B4']);self.assertEqual(state['status'],'COMPLETE_WITH_FAILURES_OR_EXCLUSIONS')
            self.assertEqual(state['failed_new_cells'],1);self.assertIn(('TRAK1',('A4',)),backend.summaries[-2:])
            self.assertEqual(backend.summaries[-1][0],'TRAK2')
    def test_infrastructure_failure_stops_before_next_cell_or_protein(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=fixture(batch);backend=FakeBackend(batch);backend.infrastructure_ids={'A4'}
            with self.assertRaises(InfrastructureError):runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A4']);self.assertGreater(backend.terminated,0)
            self.assertEqual(runner.read(batch/'state/batch_state.json')['status'],'STOPPED_INFRASTRUCTURE')
    def test_control_c_terminates_children_and_keeps_progress_for_resume(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=fixture(batch);backend=FakeBackend(batch);backend.interrupt_ids={'A4'}
            with self.assertRaises(KeyboardInterrupt):runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A4']);self.assertGreater(backend.terminated,0)
            self.assertEqual(runner.read(batch/'state/batch_state.json')['status'],'PROGRESS')
    def test_successful_summary_retry_resolves_error_but_preserves_history(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=fixture(batch);backend=FakeBackend(batch);backend.summary_fail_once=True
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(state['status'],'COMPLETE');self.assertEqual(state['summary_failures'],[])
            self.assertEqual(len(state['summary_failure_history']),1)
    def test_missing_old_summary_science_is_reported_without_running_old_cells(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=fixture(batch);backend=FakeBackend(batch);backend.missing_summary=True
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A4','A5','B4'])
            self.assertEqual(state['completed_new_cells'],3)
            self.assertEqual(state['status'],'COMPLETE_WITH_FAILURES_OR_EXCLUSIONS')
            self.assertEqual(state['summary_missing']['TRAK1'][0]['sample_id'],'old_missing')
    def test_completed_source_change_requeues_only_affected_new_cell(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=fixture(batch);backend=FakeBackend(batch)
            runner.execute_queue(manifest,data,backend,batch);backend.calls=[];backend.stamps['A5']={'source':'changed-source'}
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A5']);self.assertEqual(state['status'],'COMPLETE')
    def test_missing_published_output_is_repaired_without_repeating_cached_science(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=fixture(batch);backend=FakeBackend(batch)
            runner.execute_queue(manifest,data,backend,batch);backend.calls=[];backend.missing_outputs={'A5'}
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A5']);self.assertEqual(backend.science_calls,['A4','A5','B4'])
            self.assertEqual(state['cells']['A5']['result']['scientific_stage_calls'],0)
            self.assertEqual(state['status'],'COMPLETE')
    def test_duplicate_combined_start_is_rejected_without_removing_owner(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td)
            with runner.BatchLock(batch,FakePlatform):
                with self.assertRaises(BlockingIOError):
                    with runner.BatchLock(batch,FakePlatform):pass
                self.assertTrue((batch/'state/combined.lock/owner.json').is_file())
            self.assertFalse((batch/'state/combined.lock').exists())
    def test_queue_selector_rejects_an_old_cell_even_when_in_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);all_samples=[types.SimpleNamespace(sample_id='BR'+str(i),metadata={'BR':'BR'+str(i)}) for i in range(1,6)]
            backend=runner.SharedBackend.__new__(runner.SharedBackend);backend.common=root
            backend.manifest={'datasets':[{'molecule':'TRAK1','config_relative':'config.json','total_cells':5,'new_sample_ids':['BR1']} ]}
            backend.api=types.SimpleNamespace(load_config=lambda p:{},discover_samples=lambda c:all_samples)
            with self.assertRaisesRegex(ValueError,'Old BR1-3'):backend.load_datasets()
    def test_not_ready_manifest_cannot_start_and_installer_preserves_ordinary_launcher(self):
        with tempfile.TemporaryDirectory() as td:
            volume=Path(td);common=volume/'_TRAK_pipeline';batch=common/'batches/20261102_BR4_BR5';batch.mkdir(parents=True)
            config=batch/'config.json';config.write_text('{}')
            m=dict(schema_version=1,batch_id='20261102_BR4_BR5',common_root_relative='../..',path_basis='common_root',ready_for_execution=False,release_relative=None,new_cells=2,datasets=[dict(molecule=p,new_cells=1,new_sample_ids=[p+'_new'],config_relative='batches/20261102_BR4_BR5/config.json') for p in ('TRAK1','TRAK2')])
            manifest=batch/'batch_manifest.json';runner.atomic(manifest,m)
            with self.assertRaisesRegex(ValueError,'not been frozen'):runner.manifest_paths(manifest)
            old=volume/'START_BATCH_WINDOWS.cmd';old.write_text('ordinary launcher retained')
            result=installer.install(manifest,True)
            self.assertEqual(len(result['files']),24);self.assertEqual(old.read_text(),'ordinary launcher retained')
            text=(volume/'START_NEW_REPLICAS_WINDOWS.cmd').read_text()
            self.assertIn('replica_extension_control.ps1',text);self.assertIn('-Action "start"',text)
            self.assertNotIn('SETUP_WINDOWS',text)
            self.assertTrue((batch/'START_NEW_REPLICAS_WINDOWS.cmd').is_file())

if __name__=='__main__':unittest.main()
