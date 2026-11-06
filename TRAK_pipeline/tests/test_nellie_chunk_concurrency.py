"""Concurrent native Nellie chunks commit in chunk order with outputs identical to serial runs."""
import json,os,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import tifffile
from trak_pipeline.types import Sample,Artifact,StageContext
from trak_pipeline.stages import mask

ROOT=Path(__file__).resolve().parents[1]
FRAMES=10;CHUNK=2;SHAPE=(6,5)


def fake_native(command):
    """Stand-in for the native worker: deterministic mask from the chunk input, finishing out of order."""
    request=json.loads(Path(command[-1]).read_text(encoding='utf-8'))
    first=request['frame_ids'][0]
    time.sleep(0.05*(FRAMES-first)/CHUNK)  # later chunks finish first
    raw=tifffile.imread(request['input'])
    np.save(request['output'],raw%3==0)
    for name in ('thresholds.json','actual_parameters.json'):
        (Path(request['work'])/name).write_text(json.dumps(dict(frames=request['frame_ids'])),encoding='utf-8')


class FakePopen:
    """Popen stand-in; behaviour(command, self) runs in wait() and returns an exit code."""
    behaviour=None;started=[];lock=threading.Lock()
    def __init__(self,command,cwd=None,env=None,stdout=None,stderr=None):
        self.args=command;self.returncode=None;self.terminated=threading.Event()
        with FakePopen.lock:FakePopen.started.append(self)
    def wait(self,timeout=None):
        if self.returncode is None:
            code=FakePopen.behaviour(self.args,self)
            if self.returncode is None:self.returncode=0 if code is None else code
        return self.returncode
    def poll(self):return self.returncode
    def terminate(self):self.terminated.set()
    def kill(self):self.terminated.set()


class NellieChunkConcurrencyTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'validation').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'validation',prefix='nellie_');base=Path(self.tmp.name)
        self.data=base/'data';self.common=base/'common';self.data.mkdir();self.common.mkdir()
        self.c1=self.data/'C1.tif'
        tifffile.imwrite(self.c1,(np.arange(FRAMES*SHAPE[0]*SHAPE[1])%251).astype(np.uint16).reshape(FRAMES,*SHAPE),imagej=True,metadata=dict(axes='TYX'))
        self.env=patch.dict(os.environ,{'TRAK_COMMON_ROOT':str(self.common)});self.env.start()
    def tearDown(self):
        self.env.stop();self.tmp.cleanup()

    def context(self,workers,name=None):
        out=self.data/(name or 'broad_w%d'%workers)
        config=dict(data_root=str(self.data),acquisition=dict(shape=list(SHAPE),frames=FRAMES,pixel_size_nm=117.0,frame_interval_s=0.02),
            sample_frames=[0,5],execution=dict(chunk_frames=CHUNK,nellie_chunk_workers=workers),runtime=dict(nellie_python='nellie-python'))
        params=dict(frames=FRAMES,shape=list(SHAPE),pixel_size_nm=117.0,frame_interval_s=0.02,filter_radii_um=[0.25,1.0],nellie_version='1.0.4')
        sample=Sample('WT_TEST_BR1_T1',self.data,{},self.c1,self.c1,self.c1)
        return StageContext(config,sample,params,dict(C1=Artifact('c1','C1',dict(image=str(self.c1)))),out,self.common/'releases/x',config['runtime'])

    def run_broad(self,workers):
        ctx=self.context(workers);out=ctx.output_dir
        active=[];peak=[0];lock=threading.Lock()
        def tracked(command,process):
            with lock:active.append(1);peak[0]=max(peak[0],len(active))
            try:fake_native(command)
            finally:
                with lock:active.pop()
        FakePopen.behaviour=staticmethod(tracked);FakePopen.started=[]
        with patch.object(mask.subprocess,'Popen',FakePopen),patch('builtins.print'):
            result=mask.execute_broad(ctx)
        index=json.loads(Path(result.files['index']).read_text(encoding='utf-8'))
        packed={c['path']:mask.load_packed(out/c['path']) for c in index['chunks']}
        return index,packed,peak[0]

    def test_concurrent_matches_serial_and_keeps_chunk_order(self):
        serial_index,serial,serial_peak=self.run_broad(1)
        index,parallel,peak=self.run_broad(3)
        self.assertEqual(serial_peak,1);self.assertGreater(peak,1);self.assertLessEqual(peak,3)
        self.assertEqual([c['frame_ids'] for c in index['chunks']],[list(range(s,s+CHUNK)) for s in range(0,FRAMES,CHUNK)])
        self.assertEqual(index['chunks'],serial_index['chunks'])
        for path,(masks,ids) in serial.items():
            self.assertEqual(parallel[path][1],ids);np.testing.assert_array_equal(parallel[path][0],masks)

    def test_resume_reuses_committed_chunks_in_order(self):
        self.run_broad(3)
        index,_,peak=self.run_broad(3)  # same output dir: every chunk is already committed
        self.assertEqual(peak,0,'committed chunks must not start a native process')
        self.assertEqual([c['frame_ids'][0] for c in index['chunks']],list(range(0,FRAMES,CHUNK)))

    def test_failed_chunk_stops_running_chunks_and_starts_no_more(self):
        def behaviour(command,process):
            first=json.loads(Path(command[-1]).read_text(encoding='utf-8'))['frame_ids'][0]
            if first==0:
                time.sleep(0.2);return 3  # first chunk fails after the others have started
            stopped=process.terminated.wait(20)  # others block until terminated
            return 1 if stopped else 0
        FakePopen.behaviour=staticmethod(behaviour);FakePopen.started=[]
        ctx=self.context(3,'broad_fail');began=time.monotonic()
        with patch.object(mask.subprocess,'Popen',FakePopen),patch('builtins.print'):
            with self.assertRaises(mask.subprocess.CalledProcessError):mask.execute_broad(ctx)
        self.assertLess(time.monotonic()-began,10,'running chunks must be stopped, not awaited')
        self.assertEqual(len(FakePopen.started),3,'no chunk may start after the failure')
        self.assertTrue(all(p.terminated.is_set() for p in FakePopen.started[1:]))
        self.assertFalse(any((ctx.output_dir/'broad_packed').glob('chunk_*.json')),'no chunk is committed')

    def test_worker_count_rules(self):
        self.assertEqual(mask._nellie_chunk_workers({},environ={}),1)
        self.assertEqual(mask._nellie_chunk_workers({'execution':{'nellie_chunk_workers':3}},environ={}),3)
        self.assertEqual(mask._nellie_chunk_workers({'execution':{'nellie_chunk_workers':3}},environ={'TRAK_NELLIE_WORKERS':'2'}),2)
        with self.assertRaises(ValueError):mask._nellie_chunk_workers({'execution':{'nellie_chunk_workers':0}},environ={})


if __name__=='__main__':unittest.main()
