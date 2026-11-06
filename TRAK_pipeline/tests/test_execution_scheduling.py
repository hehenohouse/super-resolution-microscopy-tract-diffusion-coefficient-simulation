"""Execution-only scheduling: CPU counts and broad/fit overlap. No science is computed."""
import tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from trak_pipeline.types import Sample,Artifact
from trak_pipeline.graph import default_specs,stage_parameters,expected_identity
from trak_pipeline import runner
from trak_pipeline.stages import localization

ROOT=Path(__file__).resolve().parents[1]


class ExecutionCountTests(unittest.TestCase):
    def test_auto_leaves_two_cpus_and_caps(self):
        config={'execution':{'fit_workers':'auto','detection_threads':'auto'}}
        self.assertEqual(localization.fit_worker_count(config,cpu_count=24,environ={}),10)
        self.assertEqual(localization.detection_thread_count(config,cpu_count=24,environ={}),12)
        self.assertEqual(localization.fit_worker_count(config,cpu_count=10,environ={}),8)
        self.assertEqual(localization.fit_worker_count(config,cpu_count=2,environ={}),1)

    def test_legacy_defaults_and_integers(self):
        self.assertEqual(localization.fit_worker_count({},cpu_count=24,environ={}),4)
        self.assertEqual(localization.detection_thread_count({},cpu_count=24,environ={}),2)
        self.assertEqual(localization.fit_worker_count({'execution':{'fit_workers':6}},cpu_count=24,environ={}),6)

    def test_environment_override(self):
        config={'execution':{'fit_workers':'auto'}}
        self.assertEqual(localization.fit_worker_count(config,cpu_count=24,environ={'TRAK_FIT_WORKERS':'3'}),3)
        self.assertEqual(localization.detection_thread_count(config,cpu_count=24,environ={'TRAK_DETECTION_THREADS':'5'}),5)

    def test_invalid(self):
        with self.assertRaises(ValueError):localization.fit_worker_count({'execution':{'fit_workers':0}},environ={})
        with self.assertRaises(ValueError):localization.fit_worker_count({'execution':{'fit_workers':'many'}},environ={})

    def test_execution_settings_do_not_change_fit_identity(self):
        def identity(execution):
            config={'methods':{},'execution':execution}
            spec={s.stage:s for s in default_specs(config)}['fit']
            inputs={d:Artifact(d,d,{})for d in spec.dependencies}
            return expected_identity(spec,stage_parameters(config,spec),inputs)
        self.assertEqual(identity({'fit_workers':4}),identity({'fit_workers':'auto','detection_threads':'auto'}))


class BroadOverlapTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'validation').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'validation',prefix='overlap_')
        root=Path(self.tmp.name);cell=root/'WT/BR1/T5';cell.mkdir(parents=True)
        self.sample=Sample('WT_TEST_BR1_T5',cell,{},cell/'C1',cell/'C2',cell/'ROI')
        self.config={'data_root':str(root),'control_root':str(root/'_batch_pipeline'),'package_root':str(ROOT),'runtime':{},'methods':{},'style':{},'execution':{'parallel_independent_stages':True}}
    def tearDown(self):self.tmp.cleanup()

    def test_fit_starts_before_broad_finishes_and_dependants_wait(self):
        sources={k:Artifact(k,k,{})for k in('C1','C2','roi')}
        reader=Mock();reader.read_sample.return_value=dict(sources);reader.reports={}
        fit_started=threading.Event();order=[];lock=threading.Lock()
        def fake_execute(config,sample,spec,params,inputs,store,counter):
            if spec.stage=='broad':
                released=fit_started.wait(10)
                with lock:order.append(('broad_done',released))
            else:
                if spec.stage=='fit':fit_started.set()
                with lock:order.append((spec.stage,None))
            return Artifact(expected_identity(spec,params,inputs),spec.stage,{},parameters=params)
        with patch.object(runner,'_execute',side_effect=fake_execute),patch.object(runner,'_ensure_roi',side_effect=lambda c,s,a,n:a):
            report,available=runner.run_sample(self.config,self.sample,reader)
        stages=[name for name,_ in order]
        self.assertIn(('broad_done',True),order,'fit must start while Nellie is still running')
        self.assertLess(stages.index('fit'),stages.index('broad_done'))
        self.assertLess(stages.index('broad_done'),stages.index('support_counts'))
        self.assertTrue(report['science_complete']);self.assertTrue(report['publication_complete'])
        self.assertEqual(sum(1 for a in report['actions'] if a['stage']=='broad'),1)





class FitPoolStopTests(unittest.TestCase):
    def test_stop_process_pool_terminates_running_workers(self):
        import concurrent.futures,multiprocessing,time
        pool=concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'))
        futures=[pool.submit(time.sleep,60) for _ in range(4)]
        deadline=time.monotonic()+30
        while len(pool._processes or {})<2 and time.monotonic()<deadline:time.sleep(0.1)
        time.sleep(1.0)
        processes=list(pool._processes.values());self.assertEqual(len(processes),2)
        began=time.monotonic();localization.stop_process_pool(pool)
        self.assertLess(time.monotonic()-began,15)
        self.assertFalse(any(p.is_alive() for p in processes))
        # futures.wait() ignores plain CANCELLED (set by shutdown(cancel_futures=True)); poll done().
        deadline=time.monotonic()+15
        while not all(f.done() for f in futures) and time.monotonic()<deadline:time.sleep(0.1)
        self.assertTrue(all(f.done() for f in futures),'every task must be cancelled or failed promptly')
        self.assertTrue(all(f.cancelled() or f.exception() is not None for f in futures),'no task may complete normally')


if __name__=='__main__':unittest.main()
