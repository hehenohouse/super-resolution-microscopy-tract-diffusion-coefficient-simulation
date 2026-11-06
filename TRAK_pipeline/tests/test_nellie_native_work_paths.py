"""Native-path portability and checkpoint fixtures; no real Nellie/cell run."""
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path,PureWindowsPath
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import tifffile
from trak_pipeline.types import Artifact,Sample,StageContext
from trak_pipeline.stages import mask
from trak_pipeline.graph import default_specs,expected_identity

PACKAGE=Path(__file__).resolve().parents[1]


def run_style_popen(side_effect):
    """Popen stand-in driven by a subprocess.run-style fake (v4 tracks native processes via Popen)."""
    class RunStylePopen:
        calls=[]
        def __init__(self,command,**kwargs):
            RunStylePopen.calls.append(command);self.args=command;self.kwargs=kwargs;self.returncode=None
        def wait(self,timeout=None):
            if self.returncode is None:
                try:side_effect(self.args,check=True,**self.kwargs);self.returncode=0
                except subprocess.CalledProcessError as error:self.returncode=error.returncode
            return self.returncode
        def poll(self):return self.returncode
        def terminate(self):pass
        def kill(self):pass
    return RunStylePopen

class NellieNativeWorkPaths(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=PACKAGE/'validation');self.root=Path(self.tmp.name)
        self.common=self.root/'_TRAK_pipeline';self.common.mkdir()
        self.dataset=self.root/'Dataset';self.dataset.mkdir()
        self.cell=self.dataset/'long_condition_replica_cell_directory';self.cell.mkdir()
        self.raw=np.arange(4*9*9,dtype=np.uint16).reshape(4,9,9)+400
        self.c1=self.cell/'C1.tif';tifffile.imwrite(self.c1,self.raw,ome=True,metadata={'axes':'TYX'})
        self.out=self.cell/'analysis_results/modular_cache/broad'/('a'*64)
        self.config={'data_root':str(self.dataset),'acquisition':{'shape':[4,9,9],'frames':4,'pixel_size_nm':117.,'frame_interval_s':.02},'sample_frames':[0,2],'execution':{'chunk_frames':2}}
        self.parameters={'filter_radii_um':[.25,1.],'nellie_version':'0.4.1'}
        self.source=Artifact('original-C1-scientific-id','C1',{'image':str(self.c1)})
        self.sample=Sample('WT_PROTEIN_BR4_T1',self.cell,{},self.c1,self.cell/'C2.tif',self.cell/'cell.roi')
        self.ctx=StageContext(self.config,self.sample,self.parameters,{'C1':self.source},self.out,self.common,{'nellie_python':'existing-nellie-python'})
        self.env=patch.dict(os.environ,{'TRAK_COMMON_ROOT':str(self.common)});self.env.start()
    def tearDown(self):self.env.stop();self.tmp.cleanup()
    def fake_native(self,command,**kwargs):
        request=json.loads(Path(command[-1]).read_text());work=Path(request['work'])
        self.assertTrue(work.is_relative_to(self.common))
        image=tifffile.imread(request['input'])
        np.save(request['output'],image>500)
        (work/'thresholds.json').write_text(json.dumps([{'local_frame':i,'original_frame':f,'intensity_threshold_ADU':450.,'frangi_threshold':.1} for i,f in enumerate(request['frame_ids'])]))
        (work/'actual_parameters.json').write_text(json.dumps({'nellie_version':'0.4.1','filter_radii_um':[.25,1.],'native_min_area':4}))
        return subprocess.CompletedProcess(command,0)
    def test_windows_long_cell_uses_short_layout(self):
        common=PureWindowsPath(r'D:\_TRAK_pipeline')
        original=PureWindowsPath(r'D:\Xinran_TRAK1\TRAK1\WT\20260902_WT_TRAK1_BR4\090226_WTbr02_TRAK1_T1f\analysis_results\modular_cache\broad')/('a'*64)/'native_work/chunk_000000/native/nellie_necessities/input.ome-TYX-T0p02_Y0p117_X0p117-ch0-t0_to_24.ome.tif'
        work,_=mask._native_work_layout(common,'Xinran_TRAK1','a'*64,'WT_TRAK1_BR4_T1',0)
        new=work/'native/nellie_necessities/input.ome-TYX-T0p02_Y0p117_X0p117-ch0-t0_to_24.ome.tif'
        self.assertGreater(len(str(original)),260)
        self.assertLess(len(str(new)),200)
        self.assertEqual(new.drive,'D:')
    def test_full_artifact_and_cell_identity_isolate_work(self):
        a,identity=mask._native_work_layout(self.common,'Dataset','a'*64,'cell1',0)
        b,_=mask._native_work_layout(self.common,'Dataset','a'*63+'b','cell1',0)
        c,_=mask._native_work_layout(self.common,'Dataset','a'*64,'cell2',0)
        d,_=mask._native_work_layout(self.common,'OtherDataset','a'*64,'cell1',0)
        e,_=mask._native_work_layout(self.common,'Dataset','a'*64,'cell1',2)
        self.assertEqual(len({a,b,c,d,e}),5)
        self.assertEqual(identity['artifact_id'],'a'*64)
    def test_work_refuses_outside_volume_and_symlink_escape(self):
        outside=self.root.parent/('outside-'+self.root.name)
        with patch.dict(os.environ,{'TRAK_COMMON_ROOT':str(outside)}):
            with self.assertRaisesRegex(ValueError,'data drive'):mask._prepare_native_work(self.ctx,0)
        self.assertFalse(outside.exists())
        target=self.root/'outside-common';target.mkdir()
        (self.common/'native_work').symlink_to(target,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,'outside'):mask._prepare_native_work(self.ctx,0)
        self.assertEqual(list(target.iterdir()),[])
    def test_owner_conflict_does_not_delete_anything(self):
        work=mask._prepare_native_work(self.ctx,0);sentinel=work/'keep.txt';sentinel.write_text('keep')
        (work.parent/'owner.json').write_text(json.dumps({'artifact_id':'another-owner'}))
        with self.assertRaisesRegex(ValueError,'identity conflict'):mask._prepare_native_work(self.ctx,0)
        self.assertEqual(sentinel.read_text(),'keep')
    def test_native_output_commits_original_cache_and_resume_uses_no_worker(self):
        parameters=deepcopy(self.parameters);inputs=deepcopy(self.ctx.inputs)
        from trak_pipeline.graph import StageSpec
        spec=StageSpec('broad','nellie_broad','1',('C1',),True)
        before=expected_identity(spec,self.parameters,self.ctx.inputs)
        native=run_style_popen(self.fake_native)
        with patch.object(mask.subprocess,'Popen',native):
            result=mask.execute_broad(self.ctx)
        self.assertEqual(len(native.calls),2)
        self.assertEqual(self.parameters,parameters);self.assertEqual(self.ctx.inputs,inputs)
        self.assertEqual(before,expected_identity(spec,self.parameters,self.ctx.inputs))
        index=json.loads(Path(result.files['index']).read_text())
        self.assertEqual(index['available_frame_ids'],[0,1,2,3])
        for entry in index['chunks']:
            path=self.out/entry['path'];self.assertTrue(path.is_relative_to(self.out))
            actual,ids=mask.load_packed(path)
            np.testing.assert_array_equal(actual,self.raw[ids]>500)
        with patch.object(mask.subprocess,'Popen',side_effect=AssertionError('must reuse packed chunks')),patch.object(mask,'_prepare_native_work',side_effect=AssertionError('must not create native workspace')):
            again=mask.execute_broad(self.ctx)
        self.assertEqual(Path(result.files['index']).read_bytes(),Path(again.files['index']).read_bytes())
    def test_failure_keeps_work_and_native_exit_code_then_retry_recovers(self):
        failed=[]
        def fail(command,**kwargs):
            req=json.loads(Path(command[-1]).read_text());failed.append(Path(req['work']))
            kwargs['stdout'].write('Synthetic native traceback: expected failure74\n');kwargs['stdout'].flush()
            raise subprocess.CalledProcessError(74,command)
        with patch.object(mask.subprocess,'Popen',run_style_popen(fail)),contextlib.redirect_stderr(io.StringIO()) as stderr:
            with self.assertRaises(subprocess.CalledProcessError) as raised:mask.execute_broad(self.ctx)
        self.assertEqual(raised.exception.returncode,74)
        self.assertIn('expected failure74',stderr.getvalue())
        self.assertTrue(failed[0].is_dir())
        sentinel=self.common/'unrelated';sentinel.mkdir();(sentinel/'keep').write_text('keep')
        with patch.object(mask.subprocess,'Popen',run_style_popen(self.fake_native)):result=mask.execute_broad(self.ctx)
        self.assertFalse(failed[0].exists());self.assertTrue((sentinel/'keep').is_file())
        self.assertTrue(Path(result.files['index']).is_file())

if __name__=='__main__':unittest.main()
