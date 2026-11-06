"""Bounded native JVM control and persistent native-ID checkpoint contracts."""
from pathlib import Path
import csv,json,tempfile,types,unittest
from unittest.mock import patch,Mock
from trak_pipeline.stages import localization as stage
from trak_pipeline.io import atomic_json,fingerprint,read_json
from trak_pipeline import runtime

ROOT=Path(__file__).resolve().parents[1]

class DetectionMemoryLifecycleTests(unittest.TestCase):
    def setUp(self):
        directory=ROOT/'validation/detection_memory_fix_20261104/tmp';directory.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=directory);self.root=Path(self.temp.name)
        self.source=self.root/'rawC2.tif';self.source.write_bytes(b'synthetic source identity, not image data')
        self.out=self.root/'stable_detection_artifact';self.out.mkdir()
        self.settings=dict(radius_um=.28,threshold=6.5,median_filter=True,subpixel_localization=True,trackmate_version='7.14.0')
        sample=types.SimpleNamespace(C2_file=self.source,metadata={'molecule':'TEST','genotype':'WT','BR':'BR1','cell':'T1'})
        self.ctx=types.SimpleNamespace(parameters=self.settings,output_dir=self.out,config=dict(data_root=str(self.root),control_root=str(self.root/'control'),acquisition=dict(frames=1000,shape=[1000,3,4],axes='TYX',pixel_size_nm=117.,frame_interval_s=.02),sample_frames=[0],execution={'chunk_frames':25}),runtime={'detector_python':'existing-detector-python'},package_root=ROOT,sample=sample,sample_id='WT_TEST_BR1_T1')
    def tearDown(self):self.temp.cleanup()
    def commit(self,start,stop,spot_id):
        path=self.out/f'frames{start:04d}_{stop-1:04d}.csv'
        row=dict(spot_id=spot_id,frame=start,time_s=start*.02,x_px=1.,y_px=2.,x_um=.117,y_um=.234,quality_log_response=9.,radius_setting_um=.28)
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=stage.NATIVE_FIELDS);writer.writeheader();writer.writerow(row)
        record=dict(source_key=path.stem,table=str(path),start=start,stop_exclusive=stop,counts=[1]+[0]*(stop-start-1),source_fingerprint=fingerprint(self.source),settings=self.settings,table_fingerprint=fingerprint(path),sample_pixels={})
        atomic_json(path.with_suffix('.json'),record)
        return path
    def test_native_ranges_never_exceed625_and_preserve_chunk_edges(self):
        self.assertEqual(stage.detection_jvm_ranges(1000,25),[(0,625),(625,1000)])
        self.assertEqual(stage.detection_jvm_ranges(2500,25),[(0,625),(625,1250),(1250,1875),(1875,2500)])
        self.assertEqual(stage.detection_jvm_ranges(53,25),[(0,53)])
        with self.assertRaises(ValueError):stage.detection_jvm_ranges(1000,626)
    def test_invalid_native_range_rejected_before_any_java_import(self):
        job=self.root/'job.json';atomic_json(job,dict(acquisition={'frames':1000},chunk_frames=25))
        with self.assertRaisesRegex(ValueError,'at most625'):
            stage.native_main(['--job',str(job),'--start','0','--stop','650'])
    def test_resume625frames_preserves_all_saved_spotIDs_and_skips_complete_JVM_ranges(self):
        old={}
        for start in range(0,625,25):
            path=self.commit(start,start+25,10000+start);old[path]=path.read_bytes();old[path.with_suffix('.json')]=path.with_suffix('.json').read_bytes()
        invocations=[]
        def invoke(config,command,log):
            a=int(command[command.index('--start')+1]);b=int(command[command.index('--stop')+1]);invocations.append((a,b))
            for start in range(a,b,25):
                if not (self.out/f'frames{start:04d}_{start+24:04d}.json').exists():self.commit(start,start+25,0)
        with patch.object(runtime,'invoke',side_effect=invoke):request,chunks=stage.run_detection_batches(self.ctx)
        self.assertEqual(invocations,[(625,1000)])
        for path,raw in old.items():self.assertEqual(path.read_bytes(),raw)
        table=[];uids=[]
        for chunk in chunks:
            with Path(chunk['table']).open(newline='') as stream:part=list(csv.DictReader(stream))
            table.extend(part);uids.extend(stage.native_detection_uid(self.ctx.sample_id,request['source_id'],chunk['source_key'],r['spot_id']) for r in part)
        self.assertEqual(len(table),40);self.assertEqual(len(set(uids)),40)
        for start in range(0,625,25):
            row=next(r for r in table if int(r['frame'])==start)
            self.assertEqual(int(row['spot_id']),10000+start)
            self.assertIn(self.ctx.sample_id+':'+request['source_id']+f':frames{start:04d}_{start+24:04d}:{10000+start}',uids)
        self.assertTrue(all(int(r['spot_id'])==0 for r in table if int(r['frame'])>=625))
        self.assertEqual(len(read_json(self.out/'native_chunks.json')),40)
        with patch.object(runtime,'invoke',side_effect=AssertionError('A completed range must not start a JVM')):
            again_request,again_chunks=stage.run_detection_batches(self.ctx)
        self.assertEqual(again_chunks,chunks)
    def test_incompatible_or_corrupt_checkpoint_is_not_silently_replaced(self):
        path=self.commit(0,25,9);before=path.read_bytes();marker=path.with_suffix('.json');value=read_json(marker);value['settings']=dict(self.settings,threshold=7.5);atomic_json(marker,value)
        with patch.object(runtime,'invoke',side_effect=AssertionError('No overwrite')):
            with self.assertRaisesRegex(ValueError,'incompatible'):stage.run_detection_batches(self.ctx)
        self.assertEqual(path.read_bytes(),before)
        value['settings']=self.settings;atomic_json(marker,value);path.write_bytes(before+b'corrupt')
        with self.assertRaisesRegex(ValueError,'incompatible'):stage.checked_detection_chunk(self.out,0,25,fingerprint(self.source),self.settings)
    def test_java_heap_failure_becomes_infrastructure_exit74(self):
        error=RuntimeError('java.lang.OutOfMemoryError: Java heap space')
        with patch.object(runtime,'terminate_children'),patch('traceback.print_exc'):
            self.assertEqual(runtime.process_main(lambda:stage._raise_native_failure(error,625,650)),74)
    def test_nonmemory_detection_error_remains_a_cell_error(self):
        error=ValueError('invalid image axes')
        with self.assertRaisesRegex(ValueError,'axes'):stage._raise_native_failure(error,0,25)
    def test_headless_image_is_closed_and_flushed_even_if_close_fails(self):
        imp=Mock();imp.close.side_effect=RuntimeError('close failed')
        with patch('builtins.print'):stage.release_detection_image(imp)
        self.assertFalse(imp.changes);imp.close.assert_called_once_with();imp.flush.assert_called_once_with()
        stage.release_detection_image(None)

if __name__=='__main__':unittest.main()
