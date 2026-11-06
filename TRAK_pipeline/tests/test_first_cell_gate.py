"""First pending-cell completion gate; stdlib fakes, no scientific workers."""
from pathlib import Path
import importlib.util
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_extension_gate_fixtures',ROOT/'tests/test_replica_extension_batch.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
runner=base.runner

class GateBackend(base.FakeBackend):
    def __init__(self,batch):
        super().__init__(batch);self.source_failures=set();self.source_calls=[];self.result_overrides={}
    def source_stamp(self,d,s):
        self.source_calls.append(s.sample_id)
        if s.sample_id in self.source_failures:raise ValueError('Synthetic source preparation failure')
        return super().source_stamp(d,s)
    def run_cell(self,d,s):
        result=super().run_cell(d,s)
        result.update(self.result_overrides.get(s.sample_id,{}))
        return result

def gated_fixture(batch):
    manifest,data=base.fixture(batch)
    manifest.update(require_first_complete_cell=True,release_relative='releases/synthetic_v1')
    return manifest,data

class FirstCellGateTests(unittest.TestCase):
    def test_success_opens_gate_and_later_individual_failure_continues(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=gated_fixture(batch);backend=GateBackend(batch);backend.failed={'A5'}
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A4','A5','B4'])
            self.assertEqual(state['first_complete_cell_verified_release'],manifest['release_relative'])
            self.assertEqual(state['first_complete_cell_verification']['sample_id'],'A4')
            self.assertEqual(state['status'],'COMPLETE_WITH_FAILURES_OR_EXCLUSIONS')
            self.assertEqual(state['cells']['A5']['status'],'failed')
            self.assertEqual(state['cells']['B4']['status'],'complete')
    def test_first_failure_stops_then_resume_retries_same_cell(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=gated_fixture(batch);backend=GateBackend(batch);backend.failed={'A4'}
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A4'])
            self.assertEqual(state['status'],'STOPPED_FIRST_CELL_FAILURE')
            self.assertNotIn('stopped_after_cell',state)
            self.assertNotIn('first_complete_cell_verified_release',state)
            self.assertEqual(state['cells']['A4']['status'],'failed')
            self.assertFalse(any(molecule=='TRAK2' for molecule,_ in backend.summaries))
            saved=runner.read(batch/'state/batch_state.json')
            self.assertEqual(saved['first_cell_gate_failure']['sample_id'],'A4')
            self.assertGreater(backend.terminated,0)
            backend.failed.clear();state=runner.execute_queue(manifest,data,backend,batch,'resume')
            self.assertEqual(backend.calls,['A4','A4','A5','B4'])
            self.assertEqual(state['status'],'COMPLETE')
            self.assertNotIn('first_cell_gate_failure',state)
    def test_source_failure_stops_before_any_cell_execution(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=gated_fixture(batch);backend=GateBackend(batch);backend.source_failures={'A4'}
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.source_calls,['A4']);self.assertEqual(backend.calls,[])
            self.assertEqual(state['status'],'STOPPED_FIRST_CELL_FAILURE')
            self.assertEqual(state['cells']['A4']['status'],'failed')
            self.assertIsNone(state['cells']['A4']['source_stamp'])
            self.assertIn('Source preparation failed',state['first_cell_gate_failure']['reason'])
            self.assertNotIn('first_complete_cell_verified_release',state)
    def test_science_and_publication_must_both_be_explicitly_complete(self):
        cases=[({'publication_complete':False},'publication_incomplete'),
               ({'science_complete':False},'failed'),
               ({'publication_complete':None},'publication_incomplete'),
               ({'status':'publication_incomplete'},'publication_incomplete')]
        for override,status in cases:
            with self.subTest(override=override),tempfile.TemporaryDirectory() as td:
                batch=Path(td);manifest,data=gated_fixture(batch);backend=GateBackend(batch)
                backend.result_overrides['A4']=override
                state=runner.execute_queue(manifest,data,backend,batch)
                self.assertEqual(backend.calls,['A4'])
                self.assertEqual(state['status'],'STOPPED_FIRST_CELL_FAILURE')
                self.assertEqual(state['cells']['A4']['status'],status)
                self.assertNotIn('first_complete_cell_verified_release',state)
                self.assertNotIn('stopped_after_cell',state)
    def test_same_release_retains_gate_and_new_release_gates_first_pending_cell(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=gated_fixture(batch);backend=GateBackend(batch);backend.stop_on='A4'
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A4']);self.assertTrue(state['stopped_after_cell'])
            self.assertEqual(state['first_complete_cell_verified_release'],manifest['release_relative'])
            backend.stop_on=None;backend.failed={'A5'}
            state=runner.execute_queue(manifest,data,backend,batch,'resume')
            self.assertEqual(backend.calls,['A4','A5','B4'])
            self.assertEqual(state['status'],'COMPLETE_WITH_FAILURES_OR_EXCLUSIONS')
            manifest['release_relative']='releases/synthetic_v2';backend.calls=[]
            state=runner.execute_queue(manifest,data,backend,batch,'resume')
            self.assertEqual(backend.calls,['A5']);self.assertEqual(state['status'],'STOPPED_FIRST_CELL_FAILURE')
            self.assertEqual(state['first_complete_cell_verified_release'],'releases/synthetic_v1')
            self.assertEqual(state['first_cell_gate_failure']['release_relative'],'releases/synthetic_v2')
            backend.failed.clear();backend.calls=[]
            state=runner.execute_queue(manifest,data,backend,batch,'resume')
            self.assertEqual(backend.calls,['A5']);self.assertEqual(state['status'],'COMPLETE')
            self.assertEqual(state['first_complete_cell_verified_release'],'releases/synthetic_v2')
            self.assertEqual(state['first_complete_cell_verification']['sample_id'],'A5')
    def test_completed_cells_skip_and_publication_repair_uses_no_new_science(self):
        with tempfile.TemporaryDirectory() as td:
            batch=Path(td);manifest,data=base.fixture(batch);backend=GateBackend(batch)
            runner.execute_queue(manifest,data,backend,batch);backend.calls=[]
            manifest.update(require_first_complete_cell=True,release_relative='releases/synthetic_v2')
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,[])
            self.assertEqual(backend.science_calls,['A4','A5','B4'])
            self.assertNotIn('first_complete_cell_verified_release',state)
            self.assertEqual(state['status'],'COMPLETE')
            backend.missing_outputs={'A5'}
            state=runner.execute_queue(manifest,data,backend,batch)
            self.assertEqual(backend.calls,['A5'])
            self.assertEqual(backend.science_calls,['A4','A5','B4'])
            self.assertEqual(state['cells']['A5']['result']['scientific_stage_calls'],0)
            self.assertEqual(state['first_complete_cell_verification']['sample_id'],'A5')
            self.assertEqual(state['status'],'COMPLETE')

if __name__=='__main__':unittest.main()
