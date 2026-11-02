"""Command boundaries and cancellation, using only bounded synthetic artifacts."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from trak_pipeline.types import Sample,Artifact
from trak_pipeline.graph import default_specs,stage_parameters,expected_identity
from trak_pipeline import runner,runtime
from trak_pipeline.cli import main

ROOT=Path(__file__).resolve().parents[1]
class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'validation',prefix='runner_')
        root=Path(self.tmp.name);cell=root/'WT/BR1/T5';cell.mkdir(parents=True)
        self.sample=Sample('WT_TEST_BR1_T5',cell,{},cell/'C1',cell/'C2',cell/'ROI')
        self.config={'data_root':str(root),'control_root':str(root/'_batch_pipeline'),'package_root':str(ROOT),'runtime':{},'methods':{},'style':{},'execution':{'parallel_independent_stages':False}}
        self.saved={k:Artifact(k,k,{})for k in('C1','C2','roi')}
        for spec in default_specs(self.config):
            params=stage_parameters(self.config,spec)
            self.saved[spec.stage]=Artifact(expected_identity(spec,params,self.saved),spec.stage,{},parameters=params)
    def tearDown(self):self.tmp.cleanup()
    def test_completed_skip_calls_no_execution(self):
        reader=Mock();reader.read_sample.return_value=self.saved;reader.reports={}
        with patch.object(runner,'_execute',side_effect=AssertionError('Science must not run')):
            report,_=runner.run_sample(self.config,self.sample,reader)
        self.assertEqual(report['scientific_stage_calls'],0);self.assertTrue(report['science_complete']);self.assertTrue(report['publication_complete'])
    def test_refresh_missing_science_never_calls_ensure_or_execution(self):
        with patch.object(runner,'available_artifacts',return_value={'C1':self.saved['C1']}),patch.object(runner,'run_sample',side_effect=AssertionError('Upstream must not run')),patch.object(runner,'_execute',side_effect=AssertionError('No incomplete publication')),patch.object(runtime,'configure'),patch('trak_pipeline.stages.summaries.refresh',side_effect=FileNotFoundError('Saved summary is missing')):
            report=runner.refresh_dataset(self.config,[self.sample])
        self.assertEqual(report['scientific_stage_calls'],0);self.assertEqual(report['new_cells_started'],0);self.assertFalse(report['complete'])
    def test_refresh_reuses_complete_science_zero_calls(self):
        with patch.object(runner,'available_artifacts',return_value=self.saved),patch.object(runtime,'configure'),patch.object(runner,'_execute',side_effect=AssertionError('Already published')),patch('trak_pipeline.stages.summaries.refresh',return_value={'status':'reused','scientific_stage_calls':0}):
            report=runner.refresh_dataset(self.config,[self.sample])
        self.assertEqual(report['scientific_stage_calls'],0);self.assertEqual(report['cells'][0]['status'],'reused')
    def test_publication_failure_preserves_science_completion(self):
        saved=dict(self.saved);saved.pop('presentation');reader=Mock();reader.read_sample.return_value=saved;reader.reports={}
        with patch.object(runner.CacheStore,'lookup',return_value=None),patch.object(runner,'_ensure_roi',side_effect=lambda c,s,a,n:a),patch.object(runner,'_execute',side_effect=RuntimeError('PNG write failed')):
            report,_=runner.run_sample(self.config,self.sample,reader)
        self.assertTrue(report['science_complete']);self.assertFalse(report['publication_complete']);self.assertEqual(report['publication_error']['stage'],'presentation')
    def test_concurrent_child_registry_cancels_all_owned_trees(self):
        a=Mock();a.poll.return_value=None;b=Mock();b.poll.return_value=0
        with runtime._CHILD_LOCK:runtime._CHILDREN.update({id(a):a,id(b):b})
        try:
            with patch.object(runtime.platform,'terminate_tree')as terminate:runtime.terminate_children()
            terminate.assert_called_once_with(a)
        finally:
            with runtime._CHILD_LOCK:runtime._CHILDREN.clear()
            runtime.reset_cancellation()
    def test_cancelled_background_spawn_cannot_escape_registry(self):
        runtime.terminate_children()
        try:
            with patch.object(runtime,'configure'),patch.object(runtime.platform,'spawn_stage',side_effect=AssertionError('Must not spawn')):
                with self.assertRaises(KeyboardInterrupt):runtime.invoke(self.config,['not-started'],Path(self.config['data_root'])/'cancel.log')
        finally:runtime.reset_cancellation()
    def test_committed_products_take_precedence_over_legacy_adapter(self):
        legacy=self.saved['statistics'];prepared=Artifact(legacy.id,'statistics',{'directory':'prepared_statistics'},{'complete_new_plot_products':True})
        reader=Mock();reader.read_sample.return_value=self.saved
        def lookup(spec,parameters,inputs):return prepared if spec.stage=='statistics'else None
        with patch.object(runner.CacheStore,'lookup',side_effect=lookup):result=runner.available_artifacts(self.config,self.sample,reader)
        self.assertTrue(result['statistics'].metadata['complete_new_plot_products'])
    def test_native_detector_unsupported_subpixel_is_explicit(self):
        from trak_pipeline.stages.localization import validate_detection_settings
        validate_detection_settings({'subpixel_localization':True,'trackmate_version':'7.14.0'})
        with self.assertRaisesRegex(ValueError,'subpixel'):validate_detection_settings({'subpixel_localization':False})
    def test_portable_saved_roi_read_is_not_scientific(self):
        root=Path(self.config['data_root']);a=self.saved['roi'];folder=self.sample.directory/'analysis_results/modular_cache/roi_geometry'/a.id;folder.mkdir(parents=True)
        files={}
        for key in('mask','vertices'):
            p=folder/(key+'.npy');p.write_bytes(b'bounded_fixture');files[key]=p.relative_to(root).as_posix()
        (folder/'geometry.json').write_text(json.dumps({'source_roi_id':a.id,'files':files,'metadata':{'pixels':4}}))
        with patch.object(runner,'_ensure_roi',side_effect=AssertionError('No raster computation')):
            result=runner._attach_cached_roi(self.config,self.sample,a)
        self.assertEqual(result.metadata['pixels'],4);self.assertTrue(Path(result.files['mask']).is_file())
    def test_infrastructure_failure_survives_process_exit(self):
        import subprocess
        def failed():raise subprocess.CalledProcessError(74,['native'])
        with patch('traceback.print_exc'):
            self.assertEqual(runtime.process_main(failed),74)
        runtime.reset_cancellation()
    def test_stop_and_resume_only_touch_new_control(self):
        stop=runner._state(self.config)/'STOP_AFTER_CURRENT_CELL';stop.parent.mkdir(parents=True);stop.write_text('stop')
        with patch.object(runtime,'configure'),patch.object(runtime,'validate_runtime'):
            with self.assertRaises(RuntimeError):runner.run_dataset(self.config,[],skip_summary=True)
            r=runner.run_dataset(self.config,[],skip_summary=True,resume=True)
        self.assertFalse(stop.exists());self.assertTrue(list(stop.parent.glob('STOP_AFTER_CURRENT_CELL.cleared_*')));self.assertEqual(r['new_cells_started'],0)
    def test_plan_is_read_only_even_with_missing_products(self):
        reader=Mock();reader.read_sample.return_value={};reader.reports={}
        with patch('trak_pipeline.cli.load_config',return_value=self.config),patch('trak_pipeline.cli.discover_samples',return_value=[self.sample]),patch('trak_pipeline.legacy.LegacyReader',return_value=reader),patch.object(runtime,'configure',side_effect=AssertionError('Plan cannot mutate runtime')),patch('builtins.print'):
            self.assertEqual(main(['plan','--config','unused']),0)
        self.assertFalse((Path(self.config['control_root'])/'modular').exists())

if __name__=='__main__':unittest.main()
