"""Read-only legacy metadata roots stay independent of extension-batch state."""
import json
from pathlib import Path
import tempfile
import unittest
from trak_pipeline.legacy import LegacyReader

class ExtensionLegacySources(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.old=self.root/'_batch_pipeline';self.new=self.old/'batches/new';self.old.mkdir();self.new.mkdir(parents=True)
        self.output=self.root/'WT/BR1/T1/analysis_results/old';self.output.mkdir(parents=True)
        state=self.old/'state';state.mkdir();self.registry=state/'completed_cells_portable.json'
        self.registry.write_text(json.dumps({'entries':{'WT_TRAK1_BR1_T1':{'complete':True,'result_directory_relative':'WT/BR1/T1/analysis_results/old'}}}))
        self.cfg={'data_root':str(self.root),'control_root':str(self.new),'legacy_control_relative':'_batch_pipeline'}
    def tearDown(self):self.tmp.cleanup()
    def test_explicit_old_registry_reused_without_copy_or_write(self):
        before=self.registry.read_bytes();reader=LegacyReader(self.cfg)
        records=reader._completed_locations()
        self.assertEqual(records['WT_TRAK1_BR1_T1']['output'],self.output)
        self.assertEqual(reader.execution_control,self.new)
        self.assertEqual(reader.control,self.old)
        self.assertEqual(self.registry.read_bytes(),before)
        self.assertEqual(list(self.new.iterdir()),[])
    def test_existing_config_control_behavior_retained(self):
        c=dict(self.cfg);c.pop('legacy_control_relative');c['control_root']=str(self.old)
        reader=LegacyReader(c)
        self.assertIn('WT_TRAK1_BR1_T1',reader._completed_locations())
        self.assertEqual(reader.control,self.old)
    def test_missing_or_escaping_explicit_root_is_rejected(self):
        with self.assertRaises(FileNotFoundError):LegacyReader(dict(self.cfg,legacy_control_relative='missing'))
        with self.assertRaises(ValueError):LegacyReader(dict(self.cfg,legacy_control_relative='../outside'))
    def test_conflicting_legacy_outputs_are_reported(self):
        other=self.root/'WT/BR1/T1/analysis_results/other';other.mkdir()
        active=self.old/'state/active_batch.json';active.write_text(json.dumps({'cells':{'WT_TRAK1_BR1_T1':{'status':'complete','output':str(other)}}}))
        with self.assertRaisesRegex(Exception,'Conflicting completed runs'):
            LegacyReader(self.cfg)._completed_locations()

if __name__=='__main__':unittest.main()
