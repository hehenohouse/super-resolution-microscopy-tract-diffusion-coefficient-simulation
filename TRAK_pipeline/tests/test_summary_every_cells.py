"""Queue summaries refresh every N completed cells (TRAK_SUMMARY_EVERY_CELLS), on a stop request and at the
end of each dataset. Synthetic orchestration only; no scientific worker is started."""
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from test_replica_extension_batch import FakeBackend, runner


def fixture(batch, trak1=5, trak2=2):
    def sample(sid):return types.SimpleNamespace(sample_id=sid,metadata={'BR':'BR4'})
    values=[]
    for protein,prefix,n in (('TRAK1','A',trak1),('TRAK2','B',trak2)):
        new=[sample(f'{prefix}{i}') for i in range(1,n+1)]
        values.append(dict(molecule=protein,new_samples=new,all_samples=new,config=dict(control_root=str(batch/protein/'isolated_control'))))
    return dict(batch_id='synthetic_every',new_cells=trak1+trak2,existing_cells=0,summary_cells=trak1+trak2),values


class SummaryEveryCellsTests(unittest.TestCase):
    def run_queue(self,every,stop_on=None):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ,{'TRAK_SUMMARY_EVERY_CELLS':str(every)}):
            batch=Path(td);manifest,data=fixture(batch);backend=FakeBackend(batch);backend.stop_on=stop_on
            state=runner.execute_queue(manifest,data,backend,batch)
            return state,[m for m,_ in backend.summaries],backend
    def test_every_two_cells_plus_dataset_end_and_final(self):
        state,summaries,backend=self.run_queue(2)
        self.assertEqual(backend.calls,['A1','A2','A3','A4','A5','B1','B2']);self.assertEqual(state['status'],'COMPLETE')
        # startup, after A2, after A4, after the last TRAK1 cell (A5); startup, after B2; final for both
        self.assertEqual(summaries,['TRAK1','TRAK1','TRAK1','TRAK1','TRAK2','TRAK2','TRAK1','TRAK2'])
    def test_one_restores_a_summary_after_every_cell(self):
        _,summaries,_=self.run_queue(1)
        self.assertEqual(summaries,['TRAK1']*6+['TRAK2']*3+['TRAK1','TRAK2'])
    def test_stop_request_refreshes_before_leaving(self):
        state,summaries,backend=self.run_queue(10,stop_on='A2')
        self.assertEqual(backend.calls,['A1','A2']);self.assertTrue(state['stopped_after_cell'])
        self.assertEqual(summaries,['TRAK1','TRAK1']);self.assertEqual(state['summaries']['TRAK1']['phase'],'after A2')
    def test_setting_values(self):
        self.assertEqual(runner.summary_every_cells({}),10)
        self.assertEqual(runner.summary_every_cells({'TRAK_SUMMARY_EVERY_CELLS':'1'}),1)
        with self.assertRaises(ValueError):runner.summary_every_cells({'TRAK_SUMMARY_EVERY_CELLS':'0'})


if __name__=='__main__':unittest.main()
