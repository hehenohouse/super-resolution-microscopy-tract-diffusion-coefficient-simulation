"""Regression for Windows Python3.11, where Path.is_mount is unsupported."""
from contextlib import redirect_stdout
import errno
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]

def load_tool(name):
    spec=importlib.util.spec_from_file_location('_mount_fix_'+name,ROOT/'tools'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

class WindowsMountPortability(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.volume=Path(self.temp.name)
        self.common=self.volume/'_TRAK_pipeline';self.common.mkdir()
        self.batch=self.common/'batches/test';self.batch.mkdir(parents=True)
        self.runner=load_tool('run_replica_extension_batch')
        self.manifest={'batch_id':'fixture','new_cells':2,'existing_cells':2,'summary_cells':4}
        self.datasets=[{'molecule':m,'new_samples':[types.SimpleNamespace(sample_id=m+'_BR4_T1')],
                       'all_samples':[object(),object()]} for m in ('TRAK1','TRAK2')]
        self.backend=types.SimpleNamespace(runtime=types.SimpleNamespace(platform=types.SimpleNamespace(install_stop_handlers=lambda:None)),load_datasets=lambda:self.datasets)
    def tearDown(self):self.temp.cleanup()
    def forbidden_path_mount(self):
        return mock.patch.object(Path,'is_mount',side_effect=NotImplementedError('Windows Python3.11'))
    def test_plan_uses_supported_mount_probe_when_Path_method_raises(self):
        out=io.StringIO()
        with self.forbidden_path_mount() as unsupported, mock.patch.object(self.runner.os.path,'ismount',return_value=True) as supported, mock.patch.object(self.runner,'manifest_paths',return_value=(self.manifest,self.common,self.batch)), mock.patch.object(self.runner,'early_environment'), mock.patch.object(self.runner,'activate_pinned',return_value=self.common/'releases/fixture'), mock.patch.object(self.runner,'SharedBackend',return_value=self.backend), redirect_stdout(out):
            code=self.runner.main(['--manifest',str(self.batch/'manifest.json'),'--action','plan'])
        self.assertEqual(code,0);unsupported.assert_not_called()
        supported.assert_called_once_with(str(self.volume))
        report=json.loads(out.getvalue());self.assertEqual(report['new_cells_started'],0);self.assertEqual(report['scientific_stage_calls'],0)
        self.assertEqual([d['run_cells'] for d in report['datasets']],[1,1])
    def test_unmounted_volume_stops_before_loading_or_starting_a_cell(self):
        with self.forbidden_path_mount() as unsupported, mock.patch.object(self.runner.os.path,'ismount',return_value=False), mock.patch.object(self.runner,'manifest_paths',return_value=(self.manifest,self.common,self.batch)), mock.patch.object(self.runner,'activate_pinned') as activate, mock.patch.object(self.runner,'SharedBackend') as backend:
            with self.assertRaisesRegex(OSError,'must be mounted'):
                self.runner.main(['--manifest',str(self.batch/'manifest.json'),'--action','plan'])
        unsupported.assert_not_called();activate.assert_not_called();backend.assert_not_called()
    def infrastructure_backend(self):
        backend=self.runner.SharedBackend.__new__(self.runner.SharedBackend)
        backend.common=self.common;backend.platform=types.SimpleNamespace(BatchInfrastructureError=type('InfrastructureError',(RuntimeError,),{}))
        backend.datasets=[{'config':{'data_root':str(self.volume)}}]
        return backend
    def test_noninfrastructure_error_on_mounted_volume_does_not_raise_Path_error(self):
        backend=self.infrastructure_backend()
        with self.forbidden_path_mount() as unsupported, mock.patch.object(self.runner.os.path,'ismount',return_value=True):
            self.assertFalse(backend.infrastructure(OSError(errno.EINVAL,'fixture cell error')))
        unsupported.assert_not_called()
    def test_volume_disconnect_is_still_infrastructure_failure(self):
        backend=self.infrastructure_backend()
        with self.forbidden_path_mount() as unsupported, mock.patch.object(self.runner.os.path,'ismount',return_value=False):
            self.assertTrue(backend.infrastructure(OSError(errno.EINVAL,'fixture disconnected')))
        unsupported.assert_not_called()
    def test_standalone_summary_mount_check_is_also_Windows_compatible(self):
        tool=load_tool('update_intra_equal_cell');data=self.volume/'data';data.mkdir();(data/'_batch_pipeline').mkdir()
        config=self.volume/'config.json';config.write_text(json.dumps({'schema_version':1,'data_root':'data','summary':{}}))
        with self.forbidden_path_mount() as unsupported, mock.patch.object(tool,'PACKAGE',self.common), mock.patch.object(tool.os.path,'ismount',return_value=True) as supported:
            # Passing the mount probe reaches the next normal config guard;
            # it does not import/execute summary science from this fixture.
            with self.assertRaisesRegex(ValueError,'corrected v2 rule'):
                tool.configure(config)
        unsupported.assert_not_called();supported.assert_called_once_with(str(self.volume))

if __name__=='__main__':unittest.main()
